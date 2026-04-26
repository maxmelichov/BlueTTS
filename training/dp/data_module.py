import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from functools import partial
from training.data.text2latent_dataset import Text2LatentDataset
from training.utils import seed_worker


def _map_speaker_id(speaker_id, spk2idx: dict | None, unknown_spk: int = 0) -> int:
    if spk2idx is None:
        return int(speaker_id)
    if speaker_id in spk2idx:
        return int(spk2idx[speaker_id])
    try:
        return int(spk2idx.get(int(speaker_id), unknown_spk))
    except (ValueError, TypeError):
        return unknown_spk


def _map_speaker_ids(speaker_ids, spk2idx: dict | None, unknown_spk: int = 0) -> list[int]:
    return [_map_speaker_id(s, spk2idx, unknown_spk) for s in speaker_ids]


def _pad_wavs_texts(wavs, texts, speaker_ids):
    max_wav_len = max(w.numel() for w in wavs)
    max_text_len = max(t.numel() for t in texts)
    batch_size = len(wavs)

    wavs_padded = torch.zeros(batch_size, 1, max_wav_len, dtype=wavs[0].dtype)
    wav_lengths = torch.empty(batch_size, dtype=torch.long)
    texts_padded = torch.zeros(batch_size, max_text_len, dtype=texts[0].dtype)
    text_masks = torch.zeros(batch_size, 1, max_text_len, dtype=torch.float32)

    for i, (w, t) in enumerate(zip(wavs, texts)):
        wl = w.numel()
        tl = t.numel()
        wavs_padded[i, 0, :wl] = w
        wav_lengths[i] = wl
        texts_padded[i, :tl] = t
        text_masks[i, 0, :tl] = 1.0

    speaker_ids_tensor = torch.tensor(speaker_ids, dtype=torch.long)
    return wavs_padded, texts_padded, text_masks, wav_lengths, speaker_ids_tensor


def collate_with_repeat_same_file(
    batch,
    sr: int = 44100,
    repeat_p: float = 0.3,
    sep_id: int | None = None,
    n_min: int = 2,
    n_max: int = 10,
    silence_sec: float = 0.2,
    spk2idx: dict | None = None,
    unknown_spk: int = 0,
    max_total_samples: int | None = None,
):
    """
    Repeat the SAME sample N times to synthesize longer sequences.

    This is kept as an optional DP data path. The default DP dataloader below still
    uses `collate_dp` because the paper setup trains on individual utterances.
    """
    assert sep_id is not None and sep_id != 0
    assert 2 <= n_min <= n_max
    assert spk2idx is not None, "Pass spk2idx via functools.partial(...)"

    batch_size = len(batch)
    num_rep = int(batch_size * repeat_p)
    num_normal = batch_size - num_rep

    wavs = [b[0].reshape(-1) for b in batch]
    texts = [b[1] for b in batch]
    speaker_ids = _map_speaker_ids([b[2] for b in batch], spk2idx, unknown_spk)

    idxs = list(range(batch_size))
    random.shuffle(idxs)

    sep_val = int(torch.tensor([sep_id], dtype=texts[0].dtype).item())
    silence_len = int(silence_sec * sr)
    silence = torch.zeros(silence_len, dtype=wavs[0].dtype) if silence_len > 0 else None

    new_wavs = []
    new_texts = []
    new_speaker_ids = []

    for i in idxs[:num_normal]:
        new_wavs.append(wavs[i])
        new_texts.append(texts[i])
        new_speaker_ids.append(speaker_ids[i])

    for _ in range(num_rep):
        idx0 = idxs[random.randrange(batch_size)]
        w0 = wavs[idx0]
        t0 = texts[idx0]
        spk = speaker_ids[idx0]

        n_repeat = random.randint(n_min, n_max)
        if max_total_samples is not None and w0.numel() > 0:
            max_repeat = max(1, max_total_samples // w0.numel())
            n_repeat = min(n_repeat, max_repeat)

        if silence is None:
            w_cat = w0.repeat(n_repeat)
        else:
            total_len = n_repeat * w0.numel() + (n_repeat - 1) * silence_len
            w_cat = torch.empty(total_len, dtype=w0.dtype)
            pos = 0
            for k in range(n_repeat):
                w_cat[pos:pos + w0.numel()] = w0
                pos += w0.numel()
                if k < n_repeat - 1:
                    w_cat[pos:pos + silence_len] = silence
                    pos += silence_len

        if n_repeat == 1:
            t_cat = t0
        else:
            total_text_len = n_repeat * t0.numel() + (n_repeat - 1)
            t_cat = torch.empty(total_text_len, dtype=t0.dtype)
            pos = 0
            for k in range(n_repeat):
                t_cat[pos:pos + t0.numel()] = t0
                pos += t0.numel()
                if k < n_repeat - 1:
                    t_cat[pos] = sep_val
                    pos += 1

        new_wavs.append(w_cat)
        new_texts.append(t_cat)
        new_speaker_ids.append(spk)

    return _pad_wavs_texts(new_wavs, new_texts, new_speaker_ids)


def collate_dp(batch, spk2idx=None, unknown_spk=0):
    """
    Simple collate for duration predictor training (paper Sec 4.2).

    No sample repetition — each utterance is treated individually.
    The DP predicts utterance-level total latent duration.

    batch items from Text2LatentDataset:
        (wav, text_ids, speaker_id, ref_wav, is_self_ref, ref_speaker_id)
    returns:
        wavs_padded [B,1,T], texts_padded [B,L], text_masks [B,1,L],
        wav_lengths [B], speaker_ids [B]
    """
    wavs = [b[0].reshape(-1) for b in batch]
    texts = [b[1] for b in batch]
    speaker_ids_raw = [b[2] for b in batch]
    speaker_ids = _map_speaker_ids(speaker_ids_raw, spk2idx, unknown_spk)
    return _pad_wavs_texts(wavs, texts, speaker_ids)

def get_dp_dataloader(metadata_path: str, batch_size: int, num_workers: int = 16):
    dataset = Text2LatentDataset(
        metadata_path,
        sample_rate=44100,
        max_wav_len=44100 * 20,
        max_text_len=800,
    )
    speaker_ids = dataset.speaker_ids
    unique_speakers, counts = np.unique(speaker_ids, return_counts=True)
    freq = dict(zip(unique_speakers, counts))
    print(f"Speaker counts: {freq}")
    try:
        spk_raw = np.array(speaker_ids, dtype=np.int64)
        uniq = np.unique(spk_raw)
        spk2idx = {int(s): int(i) for i, s in enumerate(uniq)}
    except Exception as e:
        print(f"Warning: Could not cast speaker_ids to int64 ({e}). Using raw values.")
        uniq = np.unique(speaker_ids)
        spk2idx = {s: int(i) for i, s in enumerate(uniq)}
    num_speakers = len(uniq)
    print("num_speakers mapped:", num_speakers)
    weights = np.array([1.0 / freq[s] for s in speaker_ids], dtype=np.float32).tolist()
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    collate_fn = partial(collate_dp, spk2idx=spk2idx, unknown_spk=0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=partial(seed_worker, base_seed=42)
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
    return dataloader
