import os
import json
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tensorrt as trt

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)
from _common import Style, TextProcessor, chunk_text  # noqa: E402
from _blue_vocab import text_to_indices  # noqa: E402
del _src


_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(_logger, namespace="")

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32:   torch.int32,
    trt.int64:   torch.int64,
    trt.bool:    torch.bool,
    trt.int8:    torch.int8,
}


class TRTEngine:
    def __init__(self, engine_path: str):
        runtime = trt.Runtime(_logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self._input_names  = self._names_by_mode(trt.TensorIOMode.INPUT)
        self._output_names = self._names_by_mode(trt.TensorIOMode.OUTPUT)

    def _names_by_mode(self, mode) -> List[str]:
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
                dtype = self.engine.get_tensor_dtype(name)
                torch_dtype = _TRT_TO_TORCH.get(dtype)
                if torch_dtype is None:
                    raise TypeError(f"Unsupported TRT dtype: {dtype}")
                out = torch.empty(tuple(shape), dtype=torch_dtype, device="cuda")
                bindings[i] = out.data_ptr()
                outputs[name] = out

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, bindings[i])

        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


# ─── Main class ───────────────────────────────────────────────────────────────

class LightBlueTRT:
    def __init__(
        self,
        trt_dir: str,
        config_path: str = "tts.json",
        style_json: Optional[str] = None,
        steps: int = 32,
        cfg_scale: float = 3.0,
        speed: float = 1.0,
        seed: int = 42,
        chunk_len: int = 150,
        silence_sec: float = 0.15,
        fade_duration: float = 0.02,
        renikud_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.trt_dir = trt_dir
        self.style_json = style_json
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.speed = speed
        self.seed = seed
        self.chunk_len = chunk_len
        self.silence_sec = silence_sec
        self.fade_duration = fade_duration
        self.device = device

        self._load_config(config_path)
        self._init_engines()
        self._load_stats()
        self._load_uncond()
        self._text_proc = TextProcessor(renikud_path)

    def _load_config(self, config_path: str):
        self.normalizer_scale = 1.0
        self.latent_dim = 24
        self.chunk_compress_factor = 6
        self.hop_length = 512
        self.sample_rate = 44100

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.normalizer_scale = float(cfg.get("ttl", {}).get("normalizer", {}).get("scale", self.normalizer_scale))
            self.latent_dim = int(cfg.get("ttl", {}).get("latent_dim", self.latent_dim))
            self.chunk_compress_factor = int(cfg.get("ttl", {}).get("chunk_compress_factor", self.chunk_compress_factor))
            self.sample_rate = int(cfg.get("ae", {}).get("sample_rate", self.sample_rate))
            self.hop_length = int(cfg.get("ae", {}).get("encoder", {}).get("spec_processor", {}).get("hop_length", self.hop_length))

        self.compressed_channels = self.latent_dim * self.chunk_compress_factor

    def _load_engine(self, name: str, required: bool = True) -> Optional[TRTEngine]:
        path = os.path.join(self.trt_dir, name)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"TRT engine not found: {path}")
            return None
        return TRTEngine(path)

    def _init_engines(self):
        self._ref_enc  = self._load_engine("reference_encoder.trt", required=False)
        self._text_enc = self._load_engine("text_encoder.trt")
        self._dp       = self._load_engine("duration_predictor.trt",       required=False)
        self._dp_style = self._load_engine("duration_predictor_style.trt", required=False)
        self._vf       = self._load_engine("vector_estimator.trt")
        self._vocoder  = self._load_engine("vocoder.trt")

        vf_out = set(self._vf.output_names())
        self._vf_has_denoised = "denoised_latent" in vf_out
        self._vf_has_velocity = "velocity" in vf_out
        if not (self._vf_has_denoised or self._vf_has_velocity):
            raise ValueError(f"Unsupported vector_estimator outputs: {vf_out}")

    def _load_stats(self):
        self.mean = self.std = None
        for sp in [
            os.path.join(self.trt_dir, "stats.npz"),
            "onnx_models/stats.npz",
            "stats_multilingual.pt",
        ]:
            if not os.path.exists(sp):
                continue
            if sp.endswith(".npz"):
                s = np.load(sp)
                mean = s["mean"].astype(np.float32)
                std  = s["std"].astype(np.float32)
                if mean.ndim == 1:
                    mean = mean.reshape(1, -1, 1)
                    std  = std.reshape(1, -1, 1)
                self.mean = torch.from_numpy(mean).to(self.device)
                self.std  = torch.from_numpy(std).to(self.device)
                if "normalizer_scale" in s.files:
                    self.normalizer_scale = float(s["normalizer_scale"].item() if s["normalizer_scale"].ndim == 0 else s["normalizer_scale"][0])
            else:
                s = torch.load(sp, map_location=self.device)
                self.mean = s["mean"].view(1, -1, 1).to(self.device)
                self.std  = s["std"].view(1, -1, 1).to(self.device)
            self.compressed_channels = int(self.mean.shape[1])
            print(f"[INFO] Loaded stats from {sp}")
            break

    def _load_uncond(self):
        self._u_text = self._u_ref = None
        for up in [os.path.join(self.trt_dir, "uncond.npz"), "onnx_models/uncond.npz"]:
            if not os.path.exists(up):
                continue
            u = np.load(up)
            self._u_text = torch.from_numpy(u["u_text"].astype(np.float32)).to(self.device)
            self._u_ref  = torch.from_numpy(u["u_ref"].astype(np.float32)).to(self.device)
            break
        if self._u_text is None:
            print("[WARN] uncond.npz not found — CFG will be disabled.")

    def create(self, phonemes: str, lang: str = "he") -> Tuple[np.ndarray, int]:
        chunks = chunk_text(phonemes, self.chunk_len)
        silence = np.zeros(int(self.silence_sec * self.sample_rate), dtype=np.float32)
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(self._infer_chunk(chunk, lang=lang))
            if i < len(chunks) - 1:
                parts.append(silence)
        wav = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return wav, self.sample_rate

    def synthesize(self, text: str, lang: str = "he") -> Tuple[np.ndarray, int]:
        phonemes = self._text_proc.phonemize(text, lang=lang)
        return self.create(phonemes, lang=lang)

    def _load_style_json(self, path: str):
        with open(path) as f:
            j = json.load(f)

        def _t(key) -> Optional[torch.Tensor]:
            if key not in j:
                return None
            a = np.array(j[key]["data"], dtype=np.float32)
            t = torch.from_numpy(a).to(self.device)
            return t.unsqueeze(0) if t.dim() == 2 else t

        return _t("style_ttl"), _t("style_keys"), _t("style_dp"), _t("z_ref")

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
        ref_keys   = out.get("ref_keys")
        return ref_values, ref_keys

    def _infer_chunk(self, phonemes: str, lang: str = "he") -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("stats not loaded.")

        style_ttl = style_keys = style_dp = z_ref = None
        if self.style_json:
            style_ttl, style_keys, style_dp, z_ref = self._load_style_json(self.style_json)

        if z_ref is None and style_ttl is None:
            raise ValueError("Provide style_json with z_ref or style_ttl content.")

        indices  = text_to_indices(phonemes, lang=lang)
        text_ids = torch.tensor([indices], dtype=torch.int64, device=self.device)
        text_mask = torch.ones(1, 1, len(indices), dtype=torch.float32, device=self.device)

        z_ref_norm = None
        if z_ref is not None:
            z_ref_norm = ((z_ref - self.mean) / self.std) * float(self.normalizer_scale)
            T = z_ref_norm.shape[2]
            tail = max(2, int(T * 0.05))
            z_ref_norm = z_ref_norm[:, :, : max(1, T - tail)]
            if z_ref_norm.shape[2] > 150:
                z_ref_norm = z_ref_norm[:, :, :150]

        if style_ttl is not None:
            ref_values = style_ttl
            if ref_values.dim() == 2:
                ref_values = ref_values.unsqueeze(0)
            if style_keys is not None and style_keys.dim() == 2:
                style_keys = style_keys.unsqueeze(0)
        else:
            ref_values, style_keys = self._extract_style(z_ref_norm)

        ref_keys = style_keys if style_keys is not None else ref_values

        te_in = set(self._text_enc.input_names())
        te_feed: Dict[str, torch.Tensor] = {"text_ids": text_ids}

        if "text_mask"  in te_in: te_feed["text_mask"]  = text_mask
        if "style_ttl"  in te_in: te_feed["style_ttl"]  = ref_values
        if "ref_values" in te_in: te_feed["ref_values"] = ref_values
        if "ref_keys"   in te_in: te_feed["ref_keys"]   = ref_keys

        te_out   = self._text_enc.run(te_feed)
        text_emb = te_out.get("text_emb") or next(iter(te_out.values()))

        T_lat = self._predict_duration(text_ids, text_mask, z_ref_norm, style_dp)
        latent = self._flow_matching(text_emb, ref_values, text_mask, T_lat)
        return self._decode(latent)

    def _predict_duration(
        self,
        text_ids:   torch.Tensor,
        text_mask:  torch.Tensor,
        z_ref_norm: Optional[torch.Tensor],
        style_dp:   Optional[torch.Tensor],
    ) -> int:
        T_lat = None

        if style_dp is not None and self._dp_style is not None:
            if style_dp.dim() == 2:
                style_dp = style_dp.unsqueeze(0)
            out = self._dp_style.run({"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask})
            val = float(out["duration"].sum())
            if np.isfinite(val):
                T_lat = int(np.round(val / max(self.speed, 1e-6)))

        if T_lat is None and z_ref_norm is not None and self._dp is not None:
            ref_len  = z_ref_norm.shape[2]
            ref_mask = torch.ones(1, 1, ref_len, dtype=torch.float32, device=self.device)
            out = self._dp.run({"text_ids": text_ids, "z_ref": z_ref_norm, "text_mask": text_mask, "ref_mask": ref_mask})
            val = float(out["duration"].sum())
            if np.isfinite(val):
                T_lat = int(np.round(val / max(self.speed, 1e-6)))

        if T_lat is None:
            T_lat = int(text_ids.shape[1] * 1.3)

        txt_len = int(text_mask.sum())
        T_cap   = max(20, min(txt_len * 3 + 20, 600))
        T_lat   = min(max(int(T_lat), 1), T_cap, 800)

        max_t_lat = 2048 // self.chunk_compress_factor
        T_lat = min(T_lat, max_t_lat)

        return max(10, T_lat)

    def _vf_feed(
        self,
        noisy:      torch.Tensor,
        text_emb:   torch.Tensor,
        ref_values: torch.Tensor,
        text_mask:  torch.Tensor,
        latent_mask: torch.Tensor,
        step:       int,
    ) -> Dict[str, torch.Tensor]:
        vf_in   = set(self._vf.input_names())
        total_t = torch.tensor([float(self.steps)], dtype=torch.float32, device=self.device)
        step_t  = torch.tensor([float(step)],       dtype=torch.float32, device=self.device)

        feed: Dict[str, torch.Tensor] = {"noisy_latent": noisy}

        if "text_emb"     in vf_in: feed["text_emb"]     = text_emb
        if "text_context" in vf_in: feed["text_context"] = text_emb
        if "style_ttl"    in vf_in: feed["style_ttl"]    = ref_values
        if "ref_values"   in vf_in: feed["ref_values"]   = ref_values
        if "latent_mask"  in vf_in: feed["latent_mask"]  = latent_mask
        if "text_mask"    in vf_in: feed["text_mask"]    = text_mask
        if "style_mask"   in vf_in:
            feed["style_mask"] = torch.ones(
                1, 1, ref_values.shape[1], dtype=torch.float32, device=self.device
            )
        if "current_step" in vf_in: feed["current_step"] = step_t
        if "total_step"   in vf_in: feed["total_step"]   = total_t

        return feed

    def _flow_matching(
        self,
        text_emb:   torch.Tensor,
        ref_values: torch.Tensor,
        text_mask:  torch.Tensor,
        T_lat:      int,
    ) -> torch.Tensor:
        torch.manual_seed(self.seed)
        x = torch.randn(1, self.compressed_channels, T_lat, dtype=torch.float32, device=self.device)
        latent_mask = torch.ones(1, 1, T_lat, dtype=torch.float32, device=self.device)
        total_t = torch.tensor([float(self.steps)], dtype=torch.float32, device=self.device)
        use_cfg = self.cfg_scale != 1.0 and self._u_text is not None
        if use_cfg:
            u_text_mask = torch.ones(1, 1, 1, dtype=torch.float32, device=self.device)

        for s in range(self.steps):
            feed_cond = self._vf_feed(x, text_emb, ref_values, text_mask, latent_mask, s)
            out_cond  = self._vf.run(feed_cond)

            if self._vf_has_denoised:
                cond_next = out_cond["denoised_latent"]
            else:
                cond_next = x + out_cond["velocity"] / total_t.view(1, 1, 1)

            if use_cfg:
                feed_uncond = self._vf_feed(x, self._u_text, self._u_ref, u_text_mask, latent_mask, s)
                out_uncond  = self._vf.run(feed_uncond)
                if self._vf_has_denoised:
                    uncond_next = out_uncond["denoised_latent"]
                else:
                    uncond_next = x + out_uncond["velocity"] / total_t.view(1, 1, 1)
                x = uncond_next + self.cfg_scale * (cond_next - uncond_next)
            else:
                x = cond_next

        return x

    def _apply_fade(self, wav: np.ndarray) -> np.ndarray:
        fade_samples = int(self.fade_duration * self.sample_rate)
        if fade_samples == 0 or len(wav) < 2 * fade_samples:
            return wav
        wav = wav.copy()
        wav[:fade_samples]  *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        return wav

    def _decode(self, latent: torch.Tensor) -> np.ndarray:
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

        wav = wav.squeeze().cpu().numpy()
        return self._apply_fade(wav)



def load_voice_style(style_paths: List[str], device: str = "cuda") -> Style:
    B = len(style_paths)
    with open(style_paths[0]) as f:
        first = json.load(f)

    ttl_dims = first["style_ttl"]["dims"]
    ttl = torch.zeros(B, ttl_dims[1], ttl_dims[2], dtype=torch.float32, device=device)

    dp: Optional[torch.Tensor] = None
    if "style_dp" in first:
        dp_dims = first["style_dp"]["dims"]
        dp = torch.zeros(B, dp_dims[1], dp_dims[2], dtype=torch.float32, device=device)

    for i, path in enumerate(style_paths):
        with open(path) as f:
            d = json.load(f)
        ttl[i] = torch.tensor(d["style_ttl"]["data"], dtype=torch.float32).view(ttl_dims[1], ttl_dims[2])
        if dp is not None and "style_dp" in d:
            dp[i] = torch.tensor(d["style_dp"]["data"], dtype=torch.float32).view(dp_dims[1], dp_dims[2])

    return Style(ttl=ttl, dp=dp)
