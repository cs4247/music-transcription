"""
Cached dataset that loads pre-processed chunks from disk.
Much faster than loading raw audio/MIDI files.
"""

import os
import torch
from torch.utils.data import Dataset
import pickle

class CachedMaestroDataset(Dataset):
    """
    Dataset that loads pre-processed chunks from cache.

    Usage:
        1. First run: python preprocess_dataset.py
        2. Then use this dataset for training (10-50x faster!)

    Example:
        dataset = CachedMaestroDataset(
            cache_dir="cached_dataset",
            split="train"
        )
    """

    def __init__(self, cache_dir="cached_dataset", split="train"):
        """
        Load cached dataset.

        Args:
            cache_dir: Directory containing cached .pt files
            split: One of {"train", "validation", "test"}
        """
        self.cache_dir = cache_dir
        self.split = split
        self.split_cache_dir = os.path.join(cache_dir, split)

        # Load metadata
        metadata_path = os.path.join(cache_dir, f'{split}_metadata.pkl')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Cache not found at {metadata_path}. "
                f"Run preprocess_dataset.py first!"
            )

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.num_chunks = self.metadata['num_chunks']

        # Verify cache files exist
        if not os.path.exists(self.split_cache_dir):
            raise FileNotFoundError(
                f"Cache directory not found: {self.split_cache_dir}. "
                f"Run preprocess_dataset.py first!"
            )

        print(f"Loaded cached {split} dataset:")
        print(f"  Chunks: {self.num_chunks}")
        print(f"  Chunk length: {self.metadata['chunk_length']}s")
        print(f"  Cache dir: {self.split_cache_dir}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        """Load pre-processed chunk from cache (very fast!)"""
        cache_path = os.path.join(self.split_cache_dir, f'chunk_{idx:06d}.pt')

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cached chunk not found: {cache_path}. "
                f"Re-run preprocess_dataset.py"
            )

        # Load from cache (fast!)
        data = torch.load(cache_path)
        return data['mel'], data['roll']


class HybridMaestroDataset(Dataset):
    """
    Hybrid dataset that uses cache if available, otherwise loads from raw files.
    Best of both worlds - no preprocessing required, but benefits from cache if present.
    """

    def __init__(self, root_dir, cache_dir="cached_dataset", split="train",
                 chunk_length=None, overlap=0.0, **kwargs):
        """
        Try to load from cache, fall back to raw loading if not available.
        """
        self.use_cache = False

        # Try loading from cache first
        try:
            metadata_path = os.path.join(cache_dir, f'{split}_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)

                # Check if cache matches requested settings
                if (metadata.get('chunk_length') == chunk_length and
                    metadata.get('overlap') == overlap):

                    self.cached_dataset = CachedMaestroDataset(cache_dir, split)
                    self.use_cache = True
                    print(f"✓ Using cached dataset (fast mode!)")
                    return
        except:
            pass

        # Fall back to raw loading
        from data.dataset import MaestroDataset
        self.dataset = MaestroDataset(
            root_dir=root_dir,
            split=split,
            chunk_length=chunk_length,
            overlap=overlap,
            **kwargs
        )
        print(f"⚠ Using raw dataset (slow mode). Run preprocess_dataset.py for 10-50x speedup!")

    def __len__(self):
        if self.use_cache:
            return len(self.cached_dataset)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.use_cache:
            return self.cached_dataset[idx]
        return self.dataset[idx]
