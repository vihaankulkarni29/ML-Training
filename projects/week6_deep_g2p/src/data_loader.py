"""Data loading utilities for large G2P datasets."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NpyDataset(Dataset):
    """Simple Dataset wrapping numpy arrays stored on disk."""

    def __init__(self, features_path: str, labels_path: str):
        self.features_path = features_path
        self.labels_path = labels_path
        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if len(self.features) != len(self.labels):
            raise ValueError("Features and labels must have the same length")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        return x, y


def build_dataloader(features_path: str, labels_path: str, batch_size: int, shuffle: bool, num_workers: int = 0):
    dataset = NpyDataset(features_path, labels_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def split_indices(n: int, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_idx = indices[:n_val]
    test_idx = indices[n_val : n_val + n_test]
    train_idx = indices[n_val + n_test :]
    return train_idx, val_idx, test_idx


def create_splits(features_path: str, labels_path: str, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
    # Loads labels only to avoid pulling features into memory.
    labels = np.load(labels_path, mmap_mode="r")
    train_idx, val_idx, test_idx = split_indices(len(labels), val_frac, test_frac, seed)
    return train_idx, val_idx, test_idx
