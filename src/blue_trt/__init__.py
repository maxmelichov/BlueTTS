"""Blue TRT TTS — flat single-file TensorRT inference module (mirrors blue_onnx layout)."""

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorrt as trt
import torch

from ..blue_onnx import (
    BLUE_SYNTH_MAX_CHUNK_LEN,
    DEFAULT_MIXED_PACE_BLEND,
    DURATION_PACE_DPT_REF,
    TextProcessor,
    blend_duration_pace,
    chunk_text,
    text_to_indices,
)

# Keep in sync with ``src.blue_onnx._INLINE_LANG_PAIR`` (mixed inline-language detection).
_INLINE_LANG_PAIR = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)


# ─── TRT logger + engine wrapper ──────────────────────────────────────────────

_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(_LOGGER, namespace="")

_TRT_TO_TORCH = {
    trt.float32: torch.float32, trt.float16: torch.float16,
    trt.int32: torch.int32, trt.int64: torch.int64,
    trt.bool: torch.bool, trt.int8: torch.int8,
}


class TRTEngine:
    """Thin wrapper around a serialized TensorRT engine."""

    def __init__(self, engine_path: str):
        runtime = trt.Runtime(_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self._input_names = self._names(trt.TensorIOMode.INPUT)
        self._output_names = self._names(trt.TensorIOMode.OUTPUT)

    def _names(self, mode) -> List[str]:
        return [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == mode
        ]

    def input_names(self) -> List[str]:
        return self._input_names

    def output_names(self) -> List[str]:
        return self._output_names

    def run(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bindings = [0] * self.engine.num_io_tensors
        for name in self._input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")
            t = inputs[name]
            if not t.is_contiguous():
                t = t.contiguous()
            self.context.set_input_shape(name, t.shape)
            for i in range(self.engine.num_io_tensors):
                if self.engine.get_tensor_name(i) == name:
                    bindings[i] = t.data_ptr()
                    break

        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all binding shapes specified")

        outputs: Dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.context.get_tensor_shape(name)
                dtype = _TRT_TO_TORCH.get(self.engine.get_tensor_dtype(name))
                if dtype is None:
                    raise TypeError(f"Unsupported TRT dtype for {name}")
                out = torch.empty(tuple(shape), dtype=dtype, device="cuda")
                bindings[i] = out.data_ptr()
                outputs[name] = out

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), bindings[i])

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


# ─── Style container ──────────────────────────────────────────────────────────

@dataclass
class Style:
    ttl: Any
    dp: Optional[Any] = None
    keys: Optional[Any] = None
    z_ref: Optional[Any] = None


# ─── Module-level loaders ─────────────────────────────────────────────────────

def load_cfgs(config_path: str) -> dict:
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def load_engine(trt_dir: str, name: str, required: bool = True) -> Optional[TRTEngine]:
    path = os.path.join(trt_dir, name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"TRT engine not found: {path}")
        return None
    return TRTEngine(path)


def load_stats(trt_dir: str, device: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
    for sp in (
        os.path.join(trt_dir, "stats.npz"),
        "onnx_models/stats.npz",
        "stats_multilingual.pt",
    ):
        if not os.path.exists(sp):
            continue
        if sp.endswith(".npz"):
            s = np.load(sp)
            mean = s["mean"].astype(np.float32)
            std = s["std"].astype(np.float32)
            if mean.ndim == 1:
                mean = mean.reshape(1, -1, 1)
                std = std.reshape(1, -1, 1)
            m_t = torch.from_numpy(mean).to(device)
            s_t = torch.from_numpy(std).to(device)
            ns = 1.0
            if "normalizer_scale" in s.files:
                ns = float(s["normalizer_scale"].item() if s["normalizer_scale"].ndim == 0 else s["normalizer_scale"][0])
            print(f"[INFO] Loaded stats from {sp}")
            return m_t, s_t, ns
        s = torch.load(sp, map_location=device)
        return s["mean"].view(1, -1, 1).to(device), s["std"].view(1, -1, 1).to(device), 1.0
    return None, None, 1.0


def load_uncond(trt_dir: str, device: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    for up in (os.path.join(trt_dir, "uncond.npz"), "onnx_models/uncond.npz"):
        if os.path.exists(up):
            u = np.load(up)
            return (
                torch.from_numpy(u["u_text"].astype(np.float32)).to(device),
                torch.from_numpy(u["u_ref"].astype(np.float32)).to(device),
            )
    print("[WARN] uncond.npz not found — CFG will be disabled.")
    return None, None


def load_voice_style(style_paths: List[str], device: str = "cuda") -> Style:
    """Load one or more voice-style JSONs into a batched ``Style``."""
    B = len(style_paths)
    with open(style_paths[0]) as f:
        first = json.load(f)

    def _alloc(key):
        if key not in first:
            return None
        dims = first[key]["dims"]
        return torch.zeros(B, dims[1], dims[2], dtype=torch.float32, device=device)

    ttl = _alloc("style_ttl")
    dp = _alloc("style_dp")
    keys = _alloc("style_keys")
    z_ref = _alloc("z_ref")

    for i, path in enumerate(style_paths):
        with open(path) as f:
            d = json.load(f)
        for tgt, key in ((ttl, "style_ttl"), (dp, "style_dp"), (keys, "style_keys"), (z_ref, "z_ref")):
            if tgt is not None and key in d:
                dims = d[key]["dims"]
                tgt[i] = torch.tensor(d[key]["data"], dtype=torch.float32).view(dims[1], dims[2])

    return Style(ttl=ttl, dp=dp, keys=keys, z_ref=z_ref)


# ─── Helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def timer(name: str):
    import time
    t0 = time.time()
    print(f"{name}...")
    yield
    print(f"  -> {name} completed in {time.time() - t0:.2f} sec")


# ─── BlueTRT ──────────────────────────────────────────────────────────────────

class BlueTRT:
    def __init__(
        self,
        trt_dir: str,
        config_path: str = "tts.json",
        style_json: Optional[str] = None,
        steps: int = 5,
        cfg_scale: float = 3.0,
        speed: float = 1.0,
        seed: int = 42,
        chunk_len: int = BLUE_SYNTH_MAX_CHUNK_LEN,
        silence_sec: float = 0.15,
        fade_duration: float = 0.02,
        renikud_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.trt_dir = trt_dir
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.speed = speed
        self.seed = seed
        self.chunk_len = min(max(1, chunk_len), BLUE_SYNTH_MAX_CHUNK_LEN)
        self.silence_sec = silence_sec
        self.fade_duration = fade_duration
        self.device = device

        if renikud_path is None:
            for cand in ("model.onnx", os.path.join(trt_dir, "model.onnx")):
                if os.path.exists(cand):
                    renikud_path = cand
                    break

        cfgs = load_cfgs(config_path)
        ttl = cfgs.get("ttl", {}) or {}
        ae = cfgs.get("ae", {}) or {}
        spec = (ae.get("encoder", {}) or {}).get("spec_processor", {}) or {}
        self.normalizer_scale = float((ttl.get("normalizer", {}) or {}).get("scale", 1.0))
        self.latent_dim = int(ttl.get("latent_dim", 24))
        self.chunk_compress_factor = int(ttl.get("chunk_compress_factor", 6))
        self.hop_length = int(spec.get("hop_length", 512))
        self.sample_rate = int(ae.get("sample_rate", 44100))
        self.compressed_channels = self.latent_dim * self.chunk_compress_factor

        self._ref_enc = load_engine(trt_dir, "reference_encoder.trt", required=False)
        self._text_enc = load_engine(trt_dir, "text_encoder.trt")
        self._dp = load_engine(trt_dir, "duration_predictor.trt", required=False)
        self._dp_style = load_engine(trt_dir, "duration_predictor_style.trt", required=False)
        self._vf = load_engine(trt_dir, "vector_estimator.trt")
        self._vocoder = load_engine(trt_dir, "vocoder.trt")

        vf_out = set(self._vf.output_names())
        self._vf_has_denoised = "denoised_latent" in vf_out
        self._vf_has_velocity = "velocity" in vf_out
        if not (self._vf_has_denoised or self._vf_has_velocity):
            raise ValueError(f"Unsupported vector_estimator outputs: {vf_out}")

        self.mean, self.std, ns_from_stats = load_stats(trt_dir, device)
        if ns_from_stats != 1.0:
            self.normalizer_scale = ns_from_stats
        if self.mean is not None:
            self.compressed_channels = int(self.mean.shape[1])

        self._u_text, self._u_ref = load_uncond(trt_dir, device)
        self._style = load_voice_style([style_json], device) if style_json else None
        self._text_proc = TextProcessor(renikud_path)

    # ── Public API ──────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        lang: str = "he",
        cfg_scale: Optional[float] = None,
        text_is_phonemes: bool = False,
        pace_blend: Optional[float] = None,
        pace_dpt_ref: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        phonemes = text if text_is_phonemes else self._text_proc.phonemize(text, lang=lang)
        return self.create(
            phonemes,
            cfg_scale=cfg_scale,
            pace_blend=pace_blend,
            pace_dpt_ref=pace_dpt_ref,
        )

    def create(
        self,
        phonemes: str,
        cfg_scale: Optional[float] = None,
        pace_blend: Optional[float] = None,
        pace_dpt_ref: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        has_inline = _INLINE_LANG_PAIR.search(phonemes) is not None
        pace_blend_eff = (
            float(pace_blend)
            if pace_blend is not None
            else (DEFAULT_MIXED_PACE_BLEND if has_inline else 0.0)
        )
        cfg = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        chunks = chunk_text(phonemes, self.chunk_len)
        silence = np.zeros(int(self.silence_sec * self.sample_rate), dtype=np.float32)
        parts: List[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            parts.append(
                self._infer_chunk(
                    chunk,
                    cfg_scale=cfg,
                    pace_blend=pace_blend_eff,
                    pace_dpt_ref=pace_dpt_ref,
                )
            )
            if i < len(chunks) - 1:
                parts.append(silence)
        wav = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return wav, self.sample_rate

    # ── Internals ───────────────────────────────────────────────────────────

    def _extract_style(self, z_ref_norm: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._ref_enc is None:
            raise ValueError("reference_encoder.trt not loaded.")
        TARGET = 256
        B, C, T = z_ref_norm.shape
        if T < TARGET:
            z = torch.nn.functional.pad(z_ref_norm, (0, TARGET - T))
            mask = torch.zeros(B, 1, TARGET, dtype=torch.float32, device=self.device)
            mask[:, :, :T] = 1.0
        else:
            z = z_ref_norm[:, :, :TARGET]
            mask = torch.ones(B, 1, TARGET, dtype=torch.float32, device=self.device)

        inp_names = self._ref_enc.input_names()
        feed: Dict[str, torch.Tensor] = {"z_ref": z}
        if "mask" in inp_names:
            feed["mask"] = mask
        elif "ref_mask" in inp_names:
            feed["ref_mask"] = mask
        elif len(inp_names) >= 2:
            feed[inp_names[1]] = mask

        out = self._ref_enc.run(feed)
        ref_values = out.get("ref_values") or next(iter(out.values()))
        return ref_values, out.get("ref_keys")

    def _infer_chunk(
        self,
        phonemes: str,
        cfg_scale: float,
        pace_blend: float = 0.0,
        pace_dpt_ref: Optional[float] = None,
    ) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("stats not loaded.")
        if self._style is None:
            raise ValueError("Provide style_json with style_ttl or z_ref.")

        style_ttl = self._style.ttl
        style_keys = self._style.keys
        style_dp = self._style.dp
        z_ref = self._style.z_ref
        if z_ref is None and style_ttl is None:
            raise ValueError("Provide style_json with z_ref or style_ttl content.")

        text_ids = torch.tensor([text_to_indices(phonemes)], dtype=torch.int64, device=self.device)
        text_mask = torch.ones(1, 1, text_ids.shape[1], dtype=torch.float32, device=self.device)

        # Normalize + crop z_ref.
        z_ref_norm = None
        if z_ref is not None:
            z_ref_norm = ((z_ref - self.mean) / self.std) * float(self.normalizer_scale)
            T = z_ref_norm.shape[2]
            tail = max(2, int(T * 0.05))
            z_ref_norm = z_ref_norm[:, :, : max(1, T - tail)]
            if z_ref_norm.shape[2] > 150:
                z_ref_norm = z_ref_norm[:, :, :150]

        if style_ttl is not None:
            ref_values = style_ttl if style_ttl.dim() == 3 else style_ttl.unsqueeze(0)
            if style_keys is not None and style_keys.dim() == 2:
                style_keys = style_keys.unsqueeze(0)
        else:
            ref_values, style_keys = self._extract_style(z_ref_norm)

        ref_keys = style_keys if style_keys is not None else ref_values

        # Text encoder.
        te_in = set(self._text_enc.input_names())
        te_feed: Dict[str, torch.Tensor] = {"text_ids": text_ids}
        if "text_mask" in te_in: te_feed["text_mask"] = text_mask
        if "style_ttl" in te_in: te_feed["style_ttl"] = ref_values
        if "ref_values" in te_in: te_feed["ref_values"] = ref_values
        if "ref_keys" in te_in: te_feed["ref_keys"] = ref_keys
        te_out = self._text_enc.run(te_feed)
        text_emb = te_out.get("text_emb") or next(iter(te_out.values()))

        # Duration.
        T_lat = self._predict_duration(
            text_ids,
            text_mask,
            z_ref_norm,
            style_dp,
            pace_blend=pace_blend,
            pace_dpt_ref=pace_dpt_ref,
        )

        # Flow matching with optional CFG.
        latent = self._flow_matching(text_emb, ref_values, text_mask, T_lat, cfg_scale)

        # Decode.
        return self._decode(latent)

    def _predict_duration(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        z_ref_norm: Optional[torch.Tensor],
        style_dp: Optional[torch.Tensor],
        pace_blend: float = 0.0,
        pace_dpt_ref: Optional[float] = None,
    ) -> int:
        T_lat: Optional[int] = None
        ref = float(pace_dpt_ref) if pace_dpt_ref is not None else DURATION_PACE_DPT_REF
        tm_np = text_mask.detach().cpu().numpy()

        if style_dp is not None and self._dp_style is not None:
            if style_dp.dim() == 2:
                style_dp = style_dp.unsqueeze(0)
            out = self._dp_style.run({"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})
            val = float(out["duration"].sum())
            if np.isfinite(val):
                d = blend_duration_pace(
                    np.array([val], dtype=np.float32), tm_np, pace_blend, ref
                )
                val = float(d[0])
                T_lat = int(np.round(val / max(self.speed, 1e-6)))

        if T_lat is None and z_ref_norm is not None and self._dp is not None:
            ref_mask = torch.ones(1, 1, z_ref_norm.shape[2], dtype=torch.float32, device=self.device)
            out = self._dp.run({"text_ids": text_ids, "z_ref": z_ref_norm, "text_mask": text_mask, "ref_mask": ref_mask})
            val = float(out["duration"].sum())
            if np.isfinite(val):
                d = blend_duration_pace(
                    np.array([val], dtype=np.float32), tm_np, pace_blend, ref
                )
                val = float(d[0])
                T_lat = int(np.round(val / max(self.speed, 1e-6)))

        if T_lat is None:
            T_lat = int(text_ids.shape[1] * 1.3)

        txt_len = int(text_mask.sum())
        T_cap = max(20, min(txt_len * 3 + 20, 600))
        T_lat = min(max(int(T_lat), 1), T_cap, 800, 2048 // self.chunk_compress_factor)
        return max(10, T_lat)

    def _vf_feed(
        self,
        noisy: torch.Tensor,
        text_emb: torch.Tensor,
        ref_values: torch.Tensor,
        text_mask: torch.Tensor,
        latent_mask: torch.Tensor,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        vf_in = set(self._vf.input_names())
        total_t = torch.tensor([float(self.steps)], dtype=torch.float32, device=self.device)
        step_t = torch.tensor([float(step)], dtype=torch.float32, device=self.device)
        feed: Dict[str, torch.Tensor] = {"noisy_latent": noisy}
        if "text_emb" in vf_in: feed["text_emb"] = text_emb
        if "text_context" in vf_in: feed["text_context"] = text_emb
        if "style_ttl" in vf_in: feed["style_ttl"] = ref_values
        if "ref_values" in vf_in: feed["ref_values"] = ref_values
        if "latent_mask" in vf_in: feed["latent_mask"] = latent_mask
        if "text_mask" in vf_in: feed["text_mask"] = text_mask
        if "style_mask" in vf_in:
            feed["style_mask"] = torch.ones(1, 1, ref_values.shape[1], dtype=torch.float32, device=self.device)
        if "current_step" in vf_in: feed["current_step"] = step_t
        if "total_step" in vf_in: feed["total_step"] = total_t
        return feed

    def _flow_matching(
        self,
        text_emb: torch.Tensor,
        ref_values: torch.Tensor,
        text_mask: torch.Tensor,
        T_lat: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        torch.manual_seed(self.seed)
        x = torch.randn(1, self.compressed_channels, T_lat, dtype=torch.float32, device=self.device)
        latent_mask = torch.ones(1, 1, T_lat, dtype=torch.float32, device=self.device)

        use_cfg = cfg_scale != 1.0 and self._u_text is not None and self._u_ref is not None
        u_text_mask = torch.ones(1, 1, 1, dtype=torch.float32, device=self.device) if use_cfg else None

        for s in range(self.steps):
            cond = self._vf.run(self._vf_feed(x, text_emb, ref_values, text_mask, latent_mask, s))
            cond_next = cond["denoised_latent"] if self._vf_has_denoised else x + cond["velocity"] / float(self.steps)

            if use_cfg:
                uncond = self._vf.run(self._vf_feed(x, self._u_text, self._u_ref, u_text_mask, latent_mask, s))
                uncond_next = uncond["denoised_latent"] if self._vf_has_denoised else x + uncond["velocity"] / float(self.steps)
                # SupertonicTTS §3.4 CFG (linearity makes mixing denoiseds equivalent to mixing velocities).
                x = uncond_next + cfg_scale * (cond_next - uncond_next)
            else:
                x = cond_next

        return x

    def _decode(self, latent: torch.Tensor) -> np.ndarray:
        # Denormalize in 144-ch, then time-axis pixel-shuffle to 24-ch.
        z = (latent / float(self.normalizer_scale)) * self.std + self.mean
        B, _, T = z.shape
        z_dec = (
            z.view(B, self.latent_dim, self.chunk_compress_factor, T)
             .permute(0, 1, 3, 2)
             .reshape(B, self.latent_dim, T * self.chunk_compress_factor)
        )

        voc_out = self._vocoder.run({"latent": z_dec})
        wav = voc_out.get("waveform") or next(iter(voc_out.values()))

        frame_len = self.hop_length * self.chunk_compress_factor
        if wav.shape[-1] > 2 * frame_len:
            wav = wav[..., frame_len:-frame_len]

        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)

        fs = int(self.fade_duration * self.sample_rate)
        if fs and len(wav_np) >= 2 * fs:
            wav_np = wav_np.copy()
            wav_np[:fs] *= np.linspace(0.0, 1.0, fs, dtype=np.float32)
            wav_np[-fs:] *= np.linspace(1.0, 0.0, fs, dtype=np.float32)
        return wav_np
