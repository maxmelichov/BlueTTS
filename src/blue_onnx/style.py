"""ONNX-only voice-style extraction for BlueTTS."""
from __future__ import annotations

import json
import os
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

from . import Style


def _stats(a: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def style_from_payload(payload: dict[str, Any]) -> Style:
    """Convert a voice JSON payload into a runtime :class:`blue_onnx.Style`."""
    ttl = np.asarray(payload["style_ttl"]["data"], dtype=np.float32).reshape(payload["style_ttl"]["dims"])
    dp = np.asarray(payload["style_dp"]["data"], dtype=np.float32).reshape(payload["style_dp"]["dims"])
    return Style(ttl, dp)


def payload_from_style(style: Style, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a runtime :class:`blue_onnx.Style` into voice JSON payload data."""
    style_ttl = np.asarray(style.ttl, dtype=np.float32)
    style_dp = np.asarray(style.dp, dtype=np.float32)
    return {
        "style_ttl": {"data": style_ttl.tolist(), "dims": list(style_ttl.shape)},
        "style_dp": {"data": style_dp.tolist(), "dims": list(style_dp.shape)},
        "metadata": metadata or {},
    }


def _load_config(config: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        full = config
    else:
        with open(config, "r") as f:
            full = json.load(f)
    ae = full.get("ae", {})
    ttl = full.get("ttl", {})
    enc = ae.get("encoder", {})
    spec = enc.get("spec_processor", {})
    return {
        "full_config": full,
        "sample_rate": int(ae.get("sample_rate", 44100)),
        "n_fft": int(spec.get("n_fft", 2048)),
        "win_length": int(spec.get("win_length", spec.get("n_fft", 2048))),
        "hop_length": int(spec.get("hop_length", 512)),
        "n_mels": int(spec.get("n_mels", 1253)),
        "chunk_compress_factor": int(ttl.get("chunk_compress_factor", ae.get("chunk_compress_factor", 1))),
    }


def _read_wav_mono(path: str, sample_rate: int) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    if int(sr) != int(sample_rate):
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sample_rate), res_type="kaiser_best")
    return np.asarray(wav, dtype=np.float32)


def _linear_mel_features(wav: np.ndarray, *, sample_rate: int, n_fft: int, win_length: int, hop_length: int, n_mels: int) -> np.ndarray:
    spec = np.abs(
        librosa.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
    ).astype(np.float32)
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=None,
        htk=True,
        norm=None,
    ).astype(np.float32)
    mel = np.matmul(mel_basis, spec)
    log_spec = np.log(np.maximum(spec, 1e-5))
    log_mel = np.log(np.maximum(mel, 1e-5))
    return np.concatenate([log_spec, log_mel], axis=0)[None].astype(np.float32)


def _trim_reference_latents(z: np.ndarray, *, max_frames: int = 150) -> np.ndarray:
    t = z.shape[2]
    tail = max(2, int(t * 0.05))
    t_trim = max(1, t - tail)
    if t_trim < t:
        z = z[:, :, :t_trim]
    if z.shape[2] > max_frames:
        z = z[:, :, :max_frames]
    return z


def _session(path: str, providers: list[str] | None = None) -> ort.InferenceSession:
    return ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])


class VoiceStyleExtractor:
    """Extract ``Style`` from a WAV using only ONNX Runtime, soundfile, librosa, and numpy."""

    def __init__(
        self,
        onnx_dir: str = "onnx_models",
        *,
        config: str | dict[str, Any] = "config/tts.json",
        providers: list[str] | None = None,
        codec_encoder: str | None = None,
        style_encoder: str | None = None,
        duration_style_encoder: str | None = None,
    ):
        self.onnx_dir = onnx_dir
        self.cfg = _load_config(config)
        self.codec_encoder = _session(codec_encoder or os.path.join(onnx_dir, "codec_encoder.onnx"), providers)
        self.style_encoder = _session(style_encoder or os.path.join(onnx_dir, "style_encoder.onnx"), providers)
        self.duration_style_encoder = _session(
            duration_style_encoder or os.path.join(onnx_dir, "duration_style_encoder.onnx"),
            providers,
        )

    def _z_ref(self, ref_wav: str) -> np.ndarray:
        wav = _read_wav_mono(ref_wav, self.cfg["sample_rate"])
        mel = _linear_mel_features(
            wav,
            sample_rate=self.cfg["sample_rate"],
            n_fft=self.cfg["n_fft"],
            win_length=self.cfg["win_length"],
            hop_length=self.cfg["hop_length"],
            n_mels=self.cfg["n_mels"],
        )
        frames = (mel.shape[2] // self.cfg["chunk_compress_factor"]) * self.cfg["chunk_compress_factor"]
        if frames < mel.shape[2]:
            mel = mel[:, :, :frames]
        z_ref = self.codec_encoder.run(None, {"mel": mel})[0].astype(np.float32)
        if not np.isfinite(z_ref).all():
            raise RuntimeError("non-finite reference latents")
        return z_ref

    def payload_from_wav(self, ref_wav: str) -> dict[str, Any]:
        z_ref = _trim_reference_latents(self._z_ref(ref_wav))
        ref_mask = np.ones((z_ref.shape[0], 1, z_ref.shape[2]), dtype=np.float32)
        style_ttl = self.style_encoder.run(None, {"z_ref": z_ref, "ref_mask": ref_mask})[0].astype(np.float32)
        style_dp = self.duration_style_encoder.run(None, {"z_ref": z_ref, "ref_mask": ref_mask})[0].astype(np.float32)
        metadata = {
            "sr": self.cfg["sample_rate"],
            "ref_wav": os.path.abspath(ref_wav),
            "z_ref_norm_shape": list(z_ref.shape),
            "style_ttl_stats": _stats(style_ttl),
            "style_dp_stats": _stats(style_dp),
        }
        return payload_from_style(Style(style_ttl, style_dp), metadata=metadata)

    def from_wav(self, ref_wav: str) -> Style:
        """Extract a runtime ``Style`` object from a reference WAV."""
        return style_from_payload(self.payload_from_wav(ref_wav))


def export_voice_style(
    ref_wav: str,
    *,
    onnx_dir: str = "onnx_models",
    config: str | dict[str, Any] = "config/tts.json",
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Return a voice JSON payload from a reference WAV using ONNX Runtime only."""
    return VoiceStyleExtractor(onnx_dir=onnx_dir, config=config, providers=providers).payload_from_wav(ref_wav)


def style_from_wav(
    ref_wav: str,
    *,
    onnx_dir: str = "onnx_models",
    config: str | dict[str, Any] = "config/tts.json",
    providers: list[str] | None = None,
) -> Style:
    """Return a runtime ``Style`` object from a reference WAV using ONNX Runtime only."""
    return VoiceStyleExtractor(onnx_dir=onnx_dir, config=config, providers=providers).from_wav(ref_wav)
