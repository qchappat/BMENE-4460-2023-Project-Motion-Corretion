from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

import fft

NUM_SLICE = 16


class ReconSliceDataset:
    def __init__(self, data_csv: str, mode: str):
        df = pd.read_csv(data_csv)
        self.path = df[df['mode'] == mode]['path'].to_list()
        self.max = df[df['mode'] == mode]['max'].to_list()

    def __len__(self):
        return len(self.path) * NUM_SLICE

    def __getitem__(self, idx):
        path_idx, slc_idx = np.unravel_index(idx, (len(self.path), NUM_SLICE))
        path = self.path[path_idx]
        with h5py.File(path) as f:
            kykx = f['kspace'][slc_idx, :, :, :].copy().astype(np.complex64)  # (C,H,W)
        f.close()
        kyx = fft.ifft(kykx, axis=-1, norm='ortho').astype(np.complex64)
        yx = fft.ifft(kyx, axis=-2, norm='ortho').astype(np.complex64)
        kykx /= self.max[path_idx]
        kyx /= self.max[path_idx]
        yx /= self.max[path_idx]
        return kykx, kyx, yx


class ReconVolumeDataset:
    def __init__(self, data_csv: Path, mode: str, max_len: Optional[int] = None):
        df = pd.read_csv(data_csv)
        self.path = df[df['mode'] == mode]['path'].to_list()
        self.max = df[df['mode'] == mode]['max'].to_list()
        self.max_len = max_len

    def __len__(self):
        return len(self.path)if self.max_len is None else self.max_len

    def __getitem__(self, idx):
        path_idx = idx
        path = self.path[path_idx]
        with h5py.File(path) as f:
            kykx = f['kspace'][:, :, :, :].copy().astype(np.complex64)  # (S,C,H,W)
        f.close()
        kyx = fft.ifft(kykx, axis=-1, norm='ortho').astype(np.complex64)
        yx = fft.ifft(kyx, axis=-2, norm='ortho').astype(np.complex64)
        kykx /= self.max[path_idx]
        kyx /= self.max[path_idx]
        yx /= self.max[path_idx]
        return kykx, kyx, yx
