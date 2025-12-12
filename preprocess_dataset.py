"""
Pre-process and cache the entire dataset to disk.
This eliminates I/O bottleneck during training.

Run once: python preprocess_dataset.py
Then training loads cached .pt files instead of raw audio/MIDI.
"""

import os
import torch
from tqdm import tqdm
from data.dataset import MaestroDataset
import pickle

def preprocess_and_cache(
    root_dir="maestro-v3.0.0",
    cache_dir="cached_dataset",
    chunk_length=30.0,
    overlap=0.0,
    split='train'
):
    """
    Pre-process entire dataset and save to disk.

    This converts:
    - Raw audio files (slow to load/decode)
    - Raw MIDI files (slow to parse)
    - On-the-fly mel-spectrogram computation (CPU intensive)
    - On-the-fly piano roll computation (CPU intensive)

    Into:
    - Pre-computed tensors saved as .pt files (fast to load)

    Expected speedup: 10-50x during training!
    """

    print(f"Loading dataset metadata for {split} split...")
    dataset = MaestroDataset(
        root_dir=root_dir,
        split=split,
        chunk_length=chunk_length,
        overlap=overlap
    )

    print(f"Found {len(dataset)} chunks to preprocess")

    # Create cache directory
    split_cache_dir = os.path.join(cache_dir, split)
    os.makedirs(split_cache_dir, exist_ok=True)

    # Save dataset metadata
    metadata = {
        'root_dir': root_dir,
        'chunk_length': chunk_length,
        'overlap': overlap,
        'split': split,
        'num_chunks': len(dataset),
        'chunks': dataset.chunks if hasattr(dataset, 'chunks') else None,
        'sr': dataset.sr,
        'n_mels': dataset.n_mels,
        'hop_length': dataset.hop_length
    }

    with open(os.path.join(cache_dir, f'{split}_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Preprocessing {len(dataset)} chunks...")
    print(f"Cache directory: {split_cache_dir}")

    # Process all chunks
    for idx in tqdm(range(len(dataset)), desc=f"Caching {split}"):
        cache_path = os.path.join(split_cache_dir, f'chunk_{idx:06d}.pt')

        # Skip if already cached
        if os.path.exists(cache_path):
            continue

        # Load and process chunk (this is the slow part)
        mel_tensor, roll_tensor = dataset[idx]

        # Save to disk
        torch.save({
            'mel': mel_tensor,
            'roll': roll_tensor
        }, cache_path)

    print(f"\n✓ Preprocessing complete!")
    print(f"  Cached {len(dataset)} chunks to {split_cache_dir}")
    print(f"  Total size: ~{estimate_cache_size(split_cache_dir):.1f} GB")
    print(f"\nNow use CachedMaestroDataset for training (10-50x faster!)")

def estimate_cache_size(cache_dir):
    """Estimate total cache size in GB."""
    total_size = 0
    for f in os.listdir(cache_dir):
        if f.endswith('.pt'):
            total_size += os.path.getsize(os.path.join(cache_dir, f))
    return total_size / (1024**3)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess and cache MAESTRO dataset")
    parser.add_argument("--root_dir", type=str, default="maestro-v3.0.0")
    parser.add_argument("--cache_dir", type=str, default="cached_dataset")
    parser.add_argument("--chunk_length", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=0.0)
    args = parser.parse_args()

    print("=" * 60)
    print("MAESTRO Dataset Preprocessing")
    print("=" * 60)
    print(f"Source: {args.root_dir}")
    print(f"Cache: {args.cache_dir}")
    print(f"Chunk length: {args.chunk_length}s")
    print(f"Overlap: {args.overlap * 100:.0f}%")
    print("=" * 60)
    print()

    # Preprocess all splits
    for split in ['train', 'validation', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}\n")

        try:
            preprocess_and_cache(
                root_dir=args.root_dir,
                cache_dir=args.cache_dir,
                chunk_length=args.chunk_length,
                overlap=args.overlap,
                split=split
            )
        except Exception as e:
            print(f"Warning: Could not process {split} split: {e}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nTotal cache size: ~{estimate_cache_size(args.cache_dir):.1f} GB")
    print("\nNext steps:")
    print("1. Update your training script to use CachedMaestroDataset")
    print("2. Enjoy 10-50x faster training!")
