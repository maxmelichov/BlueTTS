"""Rewards for FlowTTS-GRPO (arXiv 2606.23190, Sec 2.3): speaker similarity, ASR error rate and DNSMOS.

Reward models are imported when built: `pip install speechbrain speechmos onnxruntime transformers`.
The ASR reward is any callable `asr(wav, sr, text, lang) -> error rate in [0, 1]`.
"""

import os

import numpy as np
import torch
import torchaudio


def _to_16k(wav, sr, kaiser=False):
    """kaiser: the training resampler (sinc, Kaiser window, width 64), used for the speaker and DNSMOS rewards."""
    w = torch.as_tensor(np.asarray(wav, dtype=np.float32))
    if w.dim() == 1:
        w = w[None]
    if sr == 16000:
        return w
    if kaiser:
        return torchaudio.functional.resample(w, sr, 16000, lowpass_filter_width=64, rolloff=0.9475937167399596,
                                              resampling_method="sinc_interp_kaiser", beta=14.769656459379492)
    return torchaudio.functional.resample(w, sr, 16000)


def error_rate(ref, hyp):
    """Levenshtein distance between two token lists over the reference length, capped at 1 (WER on words)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return min(1.0, d[-1] / max(1, len(ref)))


_BASIC = None


def words(text, english=None):
    """WER words. `english`: Whisper's English normaliser (the tokenizer's `normalize`), for English text; otherwise
    Whisper's BasicTextNormalizer without diacritics (lower case, punctuation and symbols removed), Russian ё -> е."""
    global _BASIC
    if english is not None:
        return english(text).split()
    if _BASIC is None:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        _BASIC = BasicTextNormalizer(remove_diacritics=True)
    return _BASIC(text.replace("ё", "е").replace("Ё", "Е")).split()


class SpeakerSim:
    """Cosine of ECAPA speaker embeddings at 16 kHz (the paper uses ERes2Net, a model of the same kind)."""

    def __init__(self, device="cpu"):
        tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        from speechbrain.inference.speaker import EncoderClassifier
        # speechbrain turns TF32 on at import; the trainer's numerics stay as they were
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        self.device = device
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    run_opts={"device": device})

    @torch.no_grad()
    def embed(self, wav, sr):
        """[T] or [B, T] audio -> [B, 192] unit-norm embeddings."""
        w = _to_16k(wav, sr, kaiser=True)
        e = self.model.encode_batch(w.to(self.device)).reshape(w.shape[0], -1)
        return torch.nn.functional.normalize(e, dim=-1)

    def __call__(self, wavs, sr, ref_emb):
        """Cosine of each of `wavs` ([T] or [B, T]) to the unit-norm reference embedding. Returns [B]."""
        ref_emb = ref_emb.to(self.device)
        return np.array([float(e @ ref_emb) for e in self.embed(wavs, sr)])


class DNSMOS:
    """DNSMOS P.835 overall quality (OVRL, 1-5) at 16 kHz, on one onnxruntime thread so no pool runs next to the
    trainer."""

    def __init__(self, device="cpu"):
        import onnxruntime as ort
        from speechmos import dnsmos

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        provs = ["CPUExecutionProvider"]
        if device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
            provs = [("CUDAExecutionProvider", {"device_id": torch.device(device).index or 0}), "CPUExecutionProvider"]
        mdir = os.path.join(os.path.dirname(dnsmos.__file__), "dnsmos_models")
        inst = dnsmos.DNSMOS.__new__(dnsmos.DNSMOS)
        inst.primary_model_path = os.path.join(mdir, "sig_bak_ovr.onnx")
        inst.onnx_sess = ort.InferenceSession(inst.primary_model_path, so, providers=provs)
        inst.p808_onnx_sess = ort.InferenceSession(os.path.join(mdir, "model_v8.onnx"), so, providers=provs)
        dnsmos.dnsmos = inst                                   # dnsmos.run() reuses this instance
        self._run = dnsmos.run

    def __call__(self, wav, sr):
        # the decoder can overshoot 1.0 and DNSMOS refuses such input: peak-normalised to <= 1
        w16 = _to_16k(wav, sr, kaiser=True)[0].numpy()
        w16 = np.clip(w16 / max(1.0, float(np.abs(w16).max())), -1.0, 1.0)
        return float(self._run(w16, sr=16000)["ovrl_mos"])


class WhisperASR:
    """Word error rate of a Whisper transcript in the prompt's language against its orthographic text (the paper's
    1 - WER reward). English goes through Whisper's English normaliser, other languages through `words`."""

    def __init__(self, model="openai/whisper-large-v3", device="cpu"):
        from transformers import pipeline

        dev = torch.device(device)
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=dev,
                             torch_dtype=torch.float16 if dev.type == "cuda" else torch.float32)

    def __call__(self, wav, sr, text, lang=None):
        out = self.pipe({"raw": _to_16k(wav, sr)[0].numpy(), "sampling_rate": 16000},
                        generate_kwargs={"task": "transcribe", "num_beams": 1, **({"language": lang} if lang else {})})
        en = self.pipe.tokenizer.normalize if lang == "en" else None
        return error_rate(words(text, en), words(out["text"], en))
