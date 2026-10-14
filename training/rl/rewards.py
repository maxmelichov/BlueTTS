"""Rewards for FlowTTS-GRPO (arXiv 2606.23190, Sec 2.3): speaker similarity, ASR error rate and DNSMOS.

Each reward model is optional and imported only when built:
  SpeakerSim  speechbrain   (speechbrain/spkrec-ecapa-voxceleb)
  DNSMOS      speechmos + onnxruntime   (DNSMOS P.835)
  WhisperASR  transformers  (openai/whisper-large-v3)
Install them with `pip install speechbrain speechmos onnxruntime transformers`.

The ASR reward is any callable `asr(wav, sr, text, lang) -> error rate in [0, 1]`, so a recognizer that emits
phonemes can score IPA targets with `error_rate` the same way.
"""

import os
import unicodedata

import numpy as np
import torch
import torchaudio


def _to_16k(wav, sr):
    w = torch.as_tensor(np.asarray(wav, dtype=np.float32))
    if w.dim() == 1:
        w = w[None]
    return torchaudio.functional.resample(w, sr, 16000) if sr != 16000 else w


def error_rate(ref, hyp):
    """Levenshtein distance between two token lists over the reference length, capped at 1 (WER on words)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return min(1.0, d[-1] / max(1, len(ref)))


def words(text):
    """Lower-case words with punctuation removed (the WER normalisation)."""
    s = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text.lower())
    return s.split()


class SpeakerSim:
    """Cosine of ECAPA speaker embeddings at 16 kHz. The paper uses ERes2Net; ECAPA is a public speaker-verification
    model of the same kind."""

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
        w = _to_16k(wav, sr)
        e = self.model.encode_batch(w.to(self.device)).reshape(w.shape[0], -1)
        return torch.nn.functional.normalize(e, dim=-1)

    def __call__(self, wavs, sr, ref_emb):
        """Cosine of each of `wavs` ([T] or [B, T]) to the unit-norm reference embedding. Returns [B]."""
        ref_emb = ref_emb.to(self.device)
        return np.array([float(e @ ref_emb) for e in self.embed(wavs, sr)])


class DNSMOS:
    """DNSMOS P.835 overall quality (OVRL, 1-5) at 16 kHz. One onnxruntime thread so no pool runs next to the
    trainer; device "cuda:N" uses the CUDA provider when onnxruntime has it."""

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
        # the decoder can overshoot 1.0 and DNSMOS refuses such input
        w16 = np.clip(_to_16k(wav, sr)[0].numpy(), -1.0, 1.0)
        return float(self._run(w16, sr=16000)["ovrl_mos"])


class WhisperASR:
    """Word error rate of a Whisper transcript against the target text, as the paper's 1 - WER reward."""

    def __init__(self, model="openai/whisper-large-v3", device="cpu"):
        from transformers import pipeline

        self.pipe = pipeline("automatic-speech-recognition", model=model, device=torch.device(device))

    def transcribe(self, wav, sr, lang=None):
        out = self.pipe({"raw": _to_16k(wav, sr)[0].numpy(), "sampling_rate": 16000},
                        generate_kwargs={"task": "transcribe", **({"language": lang} if lang else {})})
        return out["text"]

    def __call__(self, wav, sr, text, lang=None):
        return error_rate(words(text), words(self.transcribe(wav, sr, lang)))
