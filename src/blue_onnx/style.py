"""ONNX-only voice-style extraction for BlueTTS."""
from __future__ import annotations

import json
import math
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
    # Keep the model tag: without it a voice cloned in memory (voice_from_wav) printed the "no model tag"
    # warning on its first synthesis although its payload carried the tag.
    return Style(ttl, dp, model=payload.get("model"), speed_scale=[scale] if scale else None)


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
        with open(config, "r", encoding="utf-8") as f:
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


def _read_wav_mono(path: str, sample_rate: int, res_type: str = "kaiser_best") -> np.ndarray:
    wav, sr = sf.read(path, always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    if int(sr) != int(sample_rate):
        try:
            wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sample_rate), res_type=res_type)
        except ModuleNotFoundError:   # "kaiser_best" needs the optional resampy package; soxr ships with librosa
            wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sample_rate), res_type="soxr_hq")
    return np.asarray(wav, dtype=np.float32)


def _resample_training(wav: np.ndarray, orig_sr: int, new_sr: int, lowpass_filter_width: int = 64,
                       rolloff: float = 0.9475937167399596, beta: float = 14.769656459379492) -> np.ndarray:
    """The resampler the BlueV3 model was trained with (windowed-sinc interpolation, Kaiser window; the same
    arithmetic as torchaudio's ``sinc_interp_kaiser``), in numpy. The reference encoder is sensitive to it: the
    style tokens of a 24 kHz clip resampled another way differ a lot (cosine ~0.8 against ~0.997 with this one)."""
    g = math.gcd(int(orig_sr), int(new_sr))
    orig, new = int(orig_sr) // g, int(new_sr) // g
    if orig == new:
        return np.asarray(wav, dtype=np.float32)
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = np.arange(-width, width + orig, dtype=np.float64) / orig
    t = (np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx[None]) * base
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.i0(beta * np.sqrt(1 - (t / lowpass_filter_width) ** 2)) / np.i0(beta)
    t *= math.pi
    kernels = np.where(t == 0, 1.0, np.sin(t) / np.where(t == 0, 1.0, t)) * window * (base / orig)   # [new, K]
    # float32 from here on, as in training: the empty band above the source's Nyquist then holds the same
    # rounding residue the model saw (it reads log-spectra, so that residue is part of its input)
    x = np.pad(np.asarray(wav, dtype=np.float32), (width, width + orig))
    frames = np.lib.stride_tricks.sliding_window_view(x, kernels.shape[1])[::orig]                   # [F, K]
    out = (frames @ kernels.astype(np.float32).T).reshape(-1)
    return out[: math.ceil(new * len(wav) / orig)]


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
    onnx_dir: str = "onnx_models",
    config: str | dict[str, Any] = "config/tts.json",
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Return a voice JSON payload from a reference WAV using ONNX Runtime only (BlueV3 or blue-onnx-v2 bundle)."""
    if os.path.exists(os.path.join(onnx_dir, "reference_encoder.onnx")):
        style = _release_style_from_wav(ref_wav, onnx_dir, providers)
        payload = payload_from_style(style, metadata={"ref_wav": os.path.basename(ref_wav)})
        if style.model:
            payload["model"] = style.model
        return payload
    return VoiceStyleExtractor(onnx_dir=onnx_dir, config=config, providers=providers).payload_from_wav(ref_wav)


def _release_style_from_wav(ref_wav: str, onnx_dir: str, providers: list[str] | None = None) -> Style:
    """BlueV3 bundles: ``reference_encoder.onnx`` maps a mono 44.1 kHz waveform straight to the style tokens.
    The clip is right-padded to a multiple of ``pad_to_multiple`` samples, as the bundle's ``tts.json`` states."""
    with open(os.path.join(onnx_dir, "tts.json"), "r", encoding="utf-8") as f:
        rel = json.load(f).get("blue_release") or {}
    ref = rel.get("reference") or {}
    sr = int(ref.get("sample_rate", 44100))
    k = int(ref.get("pad_to_multiple", 3072))
    wav, src_sr = sf.read(ref_wav, always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    if int(src_sr) != sr:
        # An upsampled clip has an empty band above its own Nyquist, and the encoder reads log-spectra: what it
        # sees there is the resampler's rounding residue, so the style tokens used to change with the resampler
        # (cosine ~0.80 between two of them). Rounding to the 16-bit grid, as any PCM WAV has, gives that band a
        # defined floor: the tokens then agree (~0.985) whichever resampler produced the clip.
        wav = _resample_training(wav, int(src_sr), sr)
        wav = (np.round(wav * 32767.0) / 32767.0).astype(np.float32)
    if len(wav) < sr:
        raise ValueError(f"reference clip is {len(wav) / sr:.2f} s: use at least 1 s of speech (8-10 s works best)")
    wav = np.pad(wav, (0, (-len(wav)) % k))[None].astype(np.float32)
    style_ttl, style_dp = _session(os.path.join(onnx_dir, "reference_encoder.onnx"), providers).run(None, {"wav": wav})
    if not (np.isfinite(style_ttl).all() and np.isfinite(style_dp).all()):
        raise RuntimeError("non-finite style tokens from the reference clip")
    return Style(style_ttl.astype(np.float32), style_dp.astype(np.float32), model=rel.get("model"))


def style_from_wav(
    ref_wav: str,
    *,
    onnx_dir: str = "onnx_models",
    config: str | dict[str, Any] = "config/tts.json",
    providers: list[str] | None = None,
) -> Style:
    """Return a runtime ``Style`` object from a reference WAV using ONNX Runtime only.

    A BlueV3 bundle (``reference_encoder.onnx`` in ``onnx_dir``) uses that graph; blue-onnx-v2 uses its
    ``codec_encoder`` / ``style_encoder`` / ``duration_style_encoder`` graphs."""
    if os.path.exists(os.path.join(onnx_dir, "reference_encoder.onnx")):
        return _release_style_from_wav(ref_wav, onnx_dir, providers)
    return VoiceStyleExtractor(onnx_dir=onnx_dir, config=config, providers=providers).from_wav(ref_wav)
