"""DP data loading: natural utterances with worker-side 5%–95% reference crops."""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.data.text2latent_dataset import Text2LatentDataset
from training.data.text_vocab import PAD_ID

PACKET_SAMPLES = 512 * 6  # hop_length * chunk_compress_factor
MAX_REF_PACKETS = 256  # build_reference_only inference cap (~17.8s)


def collate_dp(batch):
    """Collate natural utterances for DP training.

    The paper's DP reference is a random 5%–95% segment of the *input
    speech*, so the crop is sampled in the worker and only the crop is
    returned. The duration target only needs the full wav length.
    """
    wavs = [item[0].reshape(-1) for item in batch]
    texts = [item[1].reshape(-1) for item in batch]
    speaker_ids = [int(item[2]) for item in batch]

    batch_size = len(batch)
    wav_lengths = torch.tensor([wav.numel() for wav in wavs], dtype=torch.long)
    text_lengths = torch.tensor([text.numel() for text in texts], dtype=torch.long)
    max_text_len = int(text_lengths.max().item())

    crops = []
    crop_packets = []
    for wav in wavs:
        valid_packets = max(1, wav.numel() // PACKET_SAMPLES)
        min_crop = max(1, math.ceil(valid_packets * 0.05))
        max_crop = max(min_crop, math.floor(valid_packets * 0.95))
        max_crop = min(max_crop, MAX_REF_PACKETS)
        min_crop = min(min_crop, max_crop)
        crop = random.randint(min_crop, max_crop)
        start = random.randint(0, valid_packets - crop)
        crops.append(wav[start * PACKET_SAMPLES : (start + crop) * PACKET_SAMPLES])
        crop_packets.append(crop)

    max_crop_samples = max(crop_packets) * PACKET_SAMPLES
    ref_wavs = torch.zeros(batch_size, max_crop_samples, dtype=wavs[0].dtype)
    texts_padded = torch.full((batch_size, max_text_len), PAD_ID, dtype=texts[0].dtype)
    text_masks = torch.zeros(batch_size, 1, max_text_len, dtype=torch.float32)

    for idx, (crop, text) in enumerate(zip(crops, texts)):
        ref_wavs[idx, : crop.numel()] = crop
        texts_padded[idx, : text.numel()] = text
        text_masks[idx, 0, : text.numel()] = 1.0

    return (
        ref_wavs,
        torch.tensor(crop_packets, dtype=torch.long),
        texts_padded,
        text_masks,
        wav_lengths,
        torch.tensor(speaker_ids, dtype=torch.long),
    )


def build_balanced_sample_weights(
    dataset: Text2LatentDataset,
    lang_target: Optional[dict] = None,
    length_buckets: tuple = (50, 100, 150, 200, 300),
    length_power: float = 0.5,
    min_bucket_count: int = 1000,
) -> torch.Tensor:
    """Per-sample weights that flatten language and text-length skew.

    Weight is ``lang_weight * bucket_count^(-length_power)``; bucket counts
    are floored at ``min_bucket_count`` so near-empty buckets are not
    oversampled into memorization.
    """
    langs = np.asarray(dataset.langs, dtype=object)
    text_lengths = dataset.df["text"].astype(str).str.len().to_numpy()
    bucket_ids = np.digitize(text_lengths, length_buckets)

    lang_weight = np.ones(len(langs), dtype=np.float64)
    if lang_target:
        unique, counts = np.unique(langs, return_counts=True)
        count_map = dict(zip(unique.tolist(), counts.tolist()))
        total = float(len(langs))
        for lang, target_prob in lang_target.items():
            if lang in count_map:
                natural_prob = count_map[lang] / total
                lang_weight[langs == lang] = target_prob / natural_prob

    # Count (lang, bucket) pairs without pandas.
    pair_keys = np.array(
        [f"{lang}|{bucket}" for lang, bucket in zip(langs, bucket_ids)],
        dtype=object,
    )
    unique_pairs, inverse, pair_counts = np.unique(
        pair_keys, return_inverse=True, return_counts=True
    )
    counts = np.maximum(pair_counts[inverse], min_bucket_count)
    bucket_weight = counts.astype(np.float64) ** (-length_power)

    weights = lang_weight * bucket_weight
    weights /= weights.sum()

    # Report effective sampled distribution.
    print("[Balance] Effective sampling distribution (lang, bucket -> mass):")
    for key, count in zip(unique_pairs.tolist(), pair_counts.tolist()):
        mass = float(weights[pair_keys == key].sum())
        print(f"  {key}: n={count} mass={mass:.3f}")
    return torch.from_numpy(weights)


def get_dp_dataloader(
    metadata_path: str,
    batch_size: int,
    sample_rate: int = 44100,
    hop_length: int = 512,
    max_wav_sec: float = 30.0,
    max_text_len: int = 800,
    num_workers: int = 16,
    balance_sampling: bool = False,
    seed: int = 42,
    device: str = "cpu",
    packet_samples: int = PACKET_SAMPLES,
) -> DataLoader:
    if packet_samples != PACKET_SAMPLES:
        raise ValueError(
            f"Config packet size {packet_samples} != collate PACKET_SAMPLES "
            f"{PACKET_SAMPLES}; update the collate constant"
        )

    max_wav_len = int(round(max_wav_sec * sample_rate)) if max_wav_sec > 0 else None
    dataset = Text2LatentDataset(
        metadata_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        max_wav_len=max_wav_len,
        max_text_len=max_text_len if max_text_len > 0 else None,
    )
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than batch_size={batch_size}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "drop_last": True,
        "num_workers": num_workers,
        "collate_fn": collate_dp,
        "pin_memory": str(device).startswith("cuda"),
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }
    if balance_sampling:
        sample_weights = build_balanced_sample_weights(
            dataset,
            lang_target={"he": 0.4, "en": 0.3},
        )
        loader_kwargs["sampler"] = WeightedRandomSampler(
            sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    else:
        loader_kwargs["shuffle"] = True
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    dataloader = DataLoader(**loader_kwargs)
    print(f"Dataset loaded with {len(dataset)} natural utterances.")
    return dataloader
