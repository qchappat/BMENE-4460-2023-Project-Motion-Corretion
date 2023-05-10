import argparse
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Add max magnitude value.')
parser.add_argument('--input_csv', required=True, help='input info csv')
parser.add_argument('--output_csv', required=True, help='output info csv')
args = parser.parse_args()

df = pd.read_csv(args.input_csv)
max_value = list()
for p in tqdm(df['path'].to_list()):
    with h5py.File(p) as f:
        kspace = f['kspace'][:]
    ispace = np.fft.ifftshift(kspace, axes=(-2, -1)).astype(np.complex64)
    ispace = np.fft.ifft2(ispace, norm='ortho').astype(np.complex64)
    ispace = np.fft.fftshift(ispace, axes=(-2, -1)).astype(np.complex64)
    max_value.append(ispace.__abs__().mean(axis=1).max())
df['max'] = max_value
df.to_csv(args.output_csv, index=False)
