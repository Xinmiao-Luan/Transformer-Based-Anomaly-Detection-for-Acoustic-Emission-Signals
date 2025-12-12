import csv
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from scipy.io import loadmat
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# =========================
# 1) Synthetic dataset
# =========================

class SyntheticSequenceDataset(Dataset):
    """
    Simple synthetic dataset (for debugging and demo).

    Each sample:
      x: [seq_len, input_dim] float tensor
      y: binary label (0/1)
    """

    def __init__(
        self,
        num_samples: int = 2000,
        seq_len: int = 256,
        input_dim: int = 16,
        anomaly_ratio: float = 0.3,
        seed: int = 0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        self.x = torch.randn(num_samples, seq_len, input_dim, generator=g)
        self.y = torch.zeros(num_samples, dtype=torch.long)
        num_anom = int(num_samples * anomaly_ratio)
        if num_anom > 0:
            idx = torch.randperm(num_samples, generator=g)[:num_anom]
            self.x[idx] += 2.0  # shift for anomalies
            self.y[idx] = 1

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_synthetic_dataloaders(
    batch_size: int = 32,
    seq_len: int = 256,
    input_dim: int = 16,
    num_samples: int = 2000,
    train_ratio: float = 0.8,
    num_workers: int = 0,
):
    dataset = SyntheticSequenceDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        input_dim=input_dim,
    )
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


# =========================
# 2) AE spectrogram dataset
# =========================

class AEAESpectrogramDataset(Dataset):
    """
    Acoustic Emission spectrogram dataset for DED processes.

    Assumptions:
      - MATLAB exported spectrograms into .mat files.
      - Each .mat file contains:
            S: [n_freq, n_time] spectrogram magnitude
            label: scalar (0/1)
      - An index CSV exists with columns:
            file,label

    This dataset:
      - Loads S from .mat
      - Applies optional log(1 + S)
      - Transposes to [seq_len, input_dim] = [n_time, n_freq]
      - Truncates/pads seq_len to a fixed length if specified
      - Normalizes each sample (optional).
    """

    def __init__(
        self,
        root_dir: str,
        index_file: str = "index.csv",
        spec_key: str = "S",
        label_key: str = "label",
        seq_len: Optional[int] = None,
        log_amplitude: bool = True,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for AEAESpectrogramDataset (pip install scipy).")

        self.root_dir = Path(root_dir)
        self.index_path = self.root_dir / index_file
        self.spec_key = spec_key
        self.label_key = label_key
        self.seq_len = seq_len
        self.log_amplitude = log_amplitude
        self.normalize = normalize
        self.dtype = dtype

        if not self.index_path.is_file():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        self.samples: List[Tuple[Path, int]] = []
        with open(self.index_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "file" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("index.csv must have columns: file,label")

            for row in reader:
                path = self.root_dir / row["file"]
                label = int(row["label"])
                if not path.is_file():
                    raise FileNotFoundError(f"Referenced .mat file not found: {path}")
                self.samples.append((path, label))

        if len(self.samples) == 0:
            raise ValueError("No samples found in index file.")

        # Infer dimensions from first sample
        example_x, _ = self._load_item(0)
        self.input_dim = example_x.shape[-1]      # n_freq
        self.seq_len_resolved = example_x.shape[0]

    def __len__(self):
        return len(self.samples)

    def _load_item(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        mat = loadmat(path)

        if self.spec_key not in mat:
            raise KeyError(f"'{self.spec_key}' not found in {path}")
        S = mat[self.spec_key]  # expected [n_freq, n_time] or [n_time, n_freq]

        S = np.array(S, dtype=np.float32)

        # Make sure S is 2D
        if S.ndim != 2:
            raise ValueError(f"S in {path} must be 2D, got shape {S.shape}")

        n0, n1 = S.shape
        # Heuristic: assume freq dimension is smaller or comparable than time
        if n1 < n0:
            S = S.T  # make sure shape is [n_freq, n_time]
        # Now S is [n_freq, n_time]
        # Optional log amplitude
        if self.log_amplitude:
            S = np.log1p(S)

        # [n_freq, n_time] -> [n_time, n_freq]
        S = S.T

        # Truncate or pad seq_len (time dimension)
        if self.seq_len is not None:
            S = self._adjust_seq_len(S, self.seq_len)

        # Normalize per-sample
        if self.normalize:
            mean = S.mean()
            std = S.std()
            if std > 0:
                S = (S - mean) / std

        x = torch.from_numpy(S).to(self.dtype)
        y = int(label)
        return x, y

    @staticmethod
    def _adjust_seq_len(S: np.ndarray, target_len: int) -> np.ndarray:
        """Truncate or pad time dimension to target_len."""
        cur_len = S.shape[0]
        if cur_len == target_len:
            return S
        elif cur_len > target_len:
            # truncate
            return S[:target_len, :]
        else:
            # pad with zeros at the end
            pad_len = target_len - cur_len
            pad = np.zeros((pad_len, S.shape[1]), dtype=S.dtype)
            return np.concatenate([S, pad], axis=0)

    def __getitem__(self, idx):
        return self._load_item(idx)


def create_ae_dataloaders(
    root_dir: str,
    index_file: str = "index.csv",
    batch_size: int = 16,
    seq_len: Optional[int] = None,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    spec_key: str = "S",
    label_key: str = "label",
    log_amplitude: bool = True,
    normalize: bool = True,
):
    """
    Build train/val DataLoaders for AE spectrogram data.
    """
    dataset = AEAESpectrogramDataset(
        root_dir=root_dir,
        index_file=index_file,
        spec_key=spec_key,
        label_key=label_key,
        seq_len=seq_len,
        log_amplitude=log_amplitude,
        normalize=normalize,
    )

    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
