import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler

from training.data.text2latent_dataset import Text2LatentDataset, collate_text2latent
from training.t2l.cfg_utils import seed_worker


def get_dataloader(metadata_path, ae_sample_rate, batch_size, is_distributed, rank=0):
    dataset = Text2LatentDataset(
        metadata_path,
        sample_rate=ae_sample_rate,
        max_wav_len=ae_sample_rate * 20,
        max_text_len=300,
    )
    if rank == 0:
        print(f"Dataset loaded with {len(dataset)} samples.")

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        speaker_ids = dataset.speaker_ids
        unique_speakers, counts = np.unique(speaker_ids, return_counts=True)
        freq = dict(zip(unique_speakers, counts))
        print(f"Speaker counts: {freq}")

        sample_weights = np.array([1.0 / freq[sid] for sid in speaker_ids])
        sample_weights = sample_weights / sample_weights.sum()
        weights = torch.from_numpy(sample_weights).double()
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=collate_text2latent,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker
    )
    
    return dataloader, sampler
