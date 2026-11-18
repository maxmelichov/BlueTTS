"""ONNX-only voice-style extraction for BlueTTS.

Two bundle layouts are supported:

- BlueV3 release bundles (``notmax123/BlueV3-onnx``): ``reference_encoder.onnx`` takes the mono
  44.1 kHz waveform and returns ``style_ttl`` and ``style_dp`` itself.
- ``notmax123/blue-onnx-v2``: ``codec_encoder`` + ``style_encoder`` + ``duration_style_encoder``
  on a linear mel spectrogram.
"""
from __future__ import annotations

import json
import os
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

from . import Style, parse_speed_scale


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
    scale = parse_speed_scale(payload.get("speed_scale"))
    return Style(ttl, dp, model=payload.get("model"), speed_scale=[scale] if scale else None)


def payload_from_style(style: Style, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a runtime :class:`blue_onnx.Style` into voice JSON payload data."""
    style_ttl = np.asarray(style.ttl, dtype=np.float32)
    style_dp = np.asarray(style.dp, dtype=np.float32)
    payload: dict[str, Any] = {
        "style_ttl": {"data": style_ttl.tolist(), "dims": list(style_ttl.shape)},
        "style_dp": {"data": style_dp.tolist(), "dims": list(style_dp.shape)},
        "metadata": metadata or {},
    }
    if style.model is not None:
        payload["model"] = style.model
    return payload


def _load_config(config: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, dict):
        full = config
    else:
        with open(config, "r", encoding="utf-8") as f:
            full = json.load(f)
    ae = full.get("ae", {})
    ttl = full.get("ttl", {})
    enc = ae.get("encoder", {})
    spec = enc.get("spec_processor", {})
    return {
        "full_config": full,
        # BlueV3 release bundles tag their voices; a voice cloned with this bundle gets the same tag.
        "model": (full.get("blue_release") or {}).get("model"),
        "base_chunk_size": int(ae.get("base_chunk_size", 512)),
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
        # soxr ships with librosa; "kaiser_best" needs resampy, which is not a dependency.
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sample_rate), res_type="soxr_vhq")
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


# BlueV3: longer clips are cut here; the reference encoder itself reads only the first ~10 s.
MAX_REFERENCE_SECONDS = 30.0
MIN_REFERENCE_SECONDS = 0.5


def _release_reference_encoder(onnx_dir: str, providers: list[str] | None = None) -> ort.InferenceSession | None:
    """The ``reference_encoder.onnx`` session of a BlueV3 bundle (one ``wav`` input), else None."""
    path = os.path.join(onnx_dir, "reference_encoder.onnx")
    if not os.path.exists(path):
        return None
    sess = _session(path, providers)
    return sess if [i.name for i in sess.get_inputs()] == ["wav"] else None


def _default_config(onnx_dir: str, config: str | dict[str, Any] | None) -> str | dict[str, Any]:
    """The bundle's own ``tts.json`` unless one is passed, like ``load_cfgs``."""
    if config is not None:
        return config
    local = os.path.join(onnx_dir, "tts.json")
    return local if os.path.exists(local) else "config/tts.json"


class VoiceStyleExtractor:
    """Extract ``Style`` from a WAV using only ONNX Runtime, soundfile, librosa, and numpy.

    ``config`` defaults to ``<onnx_dir>/tts.json`` (else ``config/tts.json``). A BlueV3 bundle
    (``reference_encoder.onnx`` with a ``wav`` input) is used as is; otherwise the v2 graphs
    ``codec_encoder``, ``style_encoder`` and ``duration_style_encoder`` must be in ``onnx_dir``.
    """

    def __init__(
        self,
        onnx_dir: str = "bluev3_onnx/onnx",
        *,
        config: str | dict[str, Any] | None = None,
        providers: list[str] | None = None,
        codec_encoder: str | None = None,
        style_encoder: str | None = None,
        duration_style_encoder: str | None = None,
    ):
        self.onnx_dir = onnx_dir
        self.cfg = _load_config(_default_config(onnx_dir, config))
        explicit_v2 = bool(codec_encoder or style_encoder or duration_style_encoder)
        self.reference_encoder = None if explicit_v2 else _release_reference_encoder(onnx_dir, providers)
        if self.reference_encoder is not None:
            return
        missing = [
            g for g, given in (("codec_encoder", codec_encoder), ("style_encoder", style_encoder),
                               ("duration_style_encoder", duration_style_encoder))
            if not given and not os.path.exists(os.path.join(onnx_dir, f"{g}.onnx"))
        ]
        if missing:
            raise FileNotFoundError(
                f"{onnx_dir} has no reference_encoder.onnx (BlueV3) and no {', '.join(missing)}.onnx (v2): "
                "download notmax123/BlueV3-onnx and pass its onnx/ directory"
            )
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

    def _release_styles(self, ref_wav: str) -> tuple[np.ndarray, np.ndarray]:
        sr = self.cfg["sample_rate"]
        wav = _read_wav_mono(ref_wav, sr)[: int(MAX_REFERENCE_SECONDS * sr)]
        if wav.size < int(MIN_REFERENCE_SECONDS * sr):
            raise ValueError(f"reference clip is shorter than {MIN_REFERENCE_SECONDS} s: {ref_wav}")
        hop = self.cfg["base_chunk_size"] * self.cfg["chunk_compress_factor"]
        wav = np.pad(wav, (0, (-wav.size) % hop)).astype(np.float32)
        style_ttl, style_dp = self.reference_encoder.run(None, {"wav": wav[None, :]})
        return np.asarray(style_ttl, dtype=np.float32), np.asarray(style_dp, dtype=np.float32)

    def payload_from_wav(self, ref_wav: str) -> dict[str, Any]:
        if self.reference_encoder is not None:
            style_ttl, style_dp = self._release_styles(ref_wav)
            metadata = {
                "sr": self.cfg["sample_rate"],
                "ref_wav": os.path.basename(ref_wav),
                "encoder": "reference_encoder.onnx",
                "style_ttl_stats": _stats(style_ttl),
                "style_dp_stats": _stats(style_dp),
            }
            return payload_from_style(Style(style_ttl, style_dp, model=self.cfg["model"]), metadata=metadata)
        z_ref = _trim_reference_latents(self._z_ref(ref_wav))
        ref_mask = np.ones((z_ref.shape[0], 1, z_ref.shape[2]), dtype=np.float32)
        style_ttl = self.style_encoder.run(None, {"z_ref": z_ref, "ref_mask": ref_mask})[0].astype(np.float32)
        style_dp = self.duration_style_encoder.run(None, {"z_ref": z_ref, "ref_mask": ref_mask})[0].astype(np.float32)
        metadata = {
            "sr": self.cfg["sample_rate"],
            # Basename, not abspath: voice JSONs get shared and committed, and the
            # full path leaks the exporter's home directory and dataset layout.
            "ref_wav": os.path.basename(ref_wav),
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
    onnx_dir: str = "bluev3_onnx/onnx",
    config: str | dict[str, Any] | None = None,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Return a voice JSON payload from a reference WAV using ONNX Runtime only."""
    return VoiceStyleExtractor(onnx_dir=onnx_dir, config=config, providers=providers).payload_from_wav(ref_wav)


def style_from_wav(
    ref_wav: str,
    *,
    onnx_dir: str = "bluev3_onnx/onnx",
    config: str | dict[str, Any] | None = None,
    providers: list[str] | None = None,
) -> Style:
    """Return a runtime ``Style`` object from a reference WAV using ONNX Runtime only."""
    return VoiceStyleExtractor(onnx_dir=onnx_dir, config=config, providers=providers).from_wav(ref_wav)
