from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

import fft

NUM_SLICE = 16

class AccSliceDataset:
    def __init__(self, data_csv: str, mode: str, corresp_dict: dict):
        df = pd.read_csv(data_csv)
        self.path = df[df['mode'] == mode]['path'].to_list()
        self.max = df[df['mode'] == mode]['max'].to_list()
        self.df = df
        self.corresp_dict = corresp_dict

    def __len__(self):
        return len(self.path) * NUM_SLICE

    def __getitem__(self, idx):
        path_idx, slc_idx = np.unravel_index(idx, (len(self.path), NUM_SLICE))
        path = self.path[path_idx]
        with h5py.File(path, 'r') as f:
            kspace = f['kspace'][slc_idx, :, :, :].copy().astype(np.complex64)  # (C,H,W)
        f.close()
        correspond_file_path = self.corresp_dict["/"+path.replace("\\", "/")][1:]
        with h5py.File(correspond_file_path) as f:
            original_kspace = f['kspace'][slc_idx, :, :, :].copy().astype(np.complex64)
        f.close()
        ispace = fft.ifft2(kspace, norm='ortho').astype(np.complex64)
        original_ispace = fft.ifft2(original_kspace, norm='ortho').astype(np.complex64)
        kspace /= self.max[path_idx]
        original_kspace /= self.df[self.df["path"]==correspond_file_path]["max"].values[0]
        ispace /= self.max[path_idx]
        original_ispace /= self.df[self.df["path"]==correspond_file_path]["max"].values[0]
        return kspace, ispace, original_kspace, original_ispace


class AccVolumeDataset:
    def __init__(self, data_csv: Path, mode: str, corresp_dict: dict, max_len: Optional[int] = None):
        df = pd.read_csv(data_csv)
        self.path = df[df['mode'] == mode]['path'].to_list()
        self.max = df[df['mode'] == mode]['max'].to_list()
        self.df = df
        self.corresp_dict = corresp_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.path) if self.max_len is None else self.max_len

    def __getitem__(self, idx):
        path_idx = idx
        path = self.path[path_idx]
        with h5py.File(path, 'r') as f:
            kspace = f['kspace'][:, :, :, :].copy().astype(np.complex64)  # (C,H,W)
        f.close()
        correspond_file_path = self.corresp_dict["/"+path.replace("\\", "/")][1:]
        with h5py.File(correspond_file_path) as f:
            original_kspace = f['kspace'][:, :, :, :].copy().astype(np.complex64)
        f.close()
        ispace = fft.ifft2(kspace, norm='ortho').astype(np.complex64)
        original_ispace = fft.ifft2(original_kspace, norm='ortho').astype(np.complex64)
        kspace /= self.max[path_idx]
        original_kspace /= self.df[self.df["path"]==correspond_file_path]["max"].values[0]
        ispace /= self.max[path_idx]
        original_ispace /= self.df[self.df["path"]==correspond_file_path]["max"].values[0]
        return kspace, ispace, original_kspace, original_ispace
