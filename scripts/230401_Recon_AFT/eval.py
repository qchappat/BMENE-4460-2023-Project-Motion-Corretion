import argparse
import os
from pathlib import Path
import shutil
import sys

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import metrics
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from cuda import get_memory_usage
from dataset import ReconVolumeDataset
from logger import Logger
from models import AFT as Model
from plot import plot


def setup_save_dir(args) -> Path:
    save_dir = Path(args.save_dir)
    assert save_dir.is_dir()
    if (save_dir / 'figures').is_dir():
        shutil.rmtree(save_dir / 'figures')
    (save_dir / 'figures').mkdir()
    return save_dir


def setup_logger(save_dir: Path) -> None:
    if (save_dir / 'eval.log').is_file():
        (save_dir / 'eval.log').unlink()
    sys.stdout = Logger(sys.stdout, save_dir / 'eval.log')


def setup_dataloader(args) -> DataLoader:
    testloader = torch.utils.data.DataLoader(
        ReconVolumeDataset(args.data_csv, mode='multicoil_test', max_len=None),
        batch_size=None, shuffle=False, num_workers=1)
    return testloader


def setup_model(save_dir, args) -> Module:
    model = Model()
    model.load_state_dict(torch.load(save_dir / 'weights.pt'))
    model.to(args.device)
    return model


def eval(model: Module, testloader: DataLoader, save_dir: Path, args) -> pd.DataFrame:
    model.eval()
    all_df = list()
    with PdfPages(save_dir / 'eval.pdf') as pdf, tqdm(total=len(testloader), dynamic_ncols=True) as pbar, torch.no_grad():
        for kykx, kyx, yx in testloader:
            kykx = kykx.to(args.device)
            kyx = kyx.to(args.device)
            yx = yx.to(args.device)
            pred_yx = list()
            for slc_idx in range(kykx.shape[0]):
                pred_kyx_slc, pred_yx_slc = model(kykx[slc_idx:slc_idx + 1])
                pred_yx.append(pred_yx_slc)
            pred_yx = torch.cat(pred_yx)

            img = yx.abs().mean(dim=1).detach().cpu().numpy()
            pred_img = pred_yx.abs().mean(dim=1).detach().cpu().numpy()

            img_max = img.max()

            img /= img_max
            pred_img /= img_max

            pred_metrics = dict()
            pred_metrics['SSIM'] = metrics.structural_similarity(img, pred_img, data_range=1)
            pred_metrics['PSNR'] = metrics.peak_signal_noise_ratio(img, pred_img, data_range=1)
            pred_metrics['NRMSE'] = metrics.normalized_root_mse(img, pred_img)

            pred_df = pd.DataFrame.from_dict(pred_metrics, orient='index').T
            all_df.append(pred_df)

            fig = plot(img, pred_img, pred_metrics)
            fig.set_size_inches(10, 12)
            fig.savefig(
                save_dir / 'figures' / f"{pbar.n:04d}_{pred_metrics['SSIM']:.5f}.png",
                bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            pbar.update()
            if args.debug and pbar.n == 2:
                break
    return pd.concat(all_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', required=True, help='data info csv')
    parser.add_argument('--save_dir', required=True, help='saving directory')
    parser.add_argument(
        '--device', default='cpu',
        help='the device on which a torch.Tensor is or will be allocated')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    save_dir = setup_save_dir(args)
    setup_logger(save_dir)
    testloader = setup_dataloader(args)
    model = setup_model(save_dir, args)
    df = eval(model, testloader, save_dir, args)
    df.to_csv(save_dir / 'eval.csv', index=False)

    pdf_size = (save_dir / 'eval.pdf').stat().st_size
    fig_size = sum(f.stat().st_size for f in (save_dir / 'figures').glob('**/*.png') if f.is_file())

    s = f'pid = {os.getpid()}\n'
    s += f'GPU_memory_usage = {get_memory_usage()}\n'
    for c in ['SSIM', 'PSNR', 'NRMSE']:
        s += f"Input_{c} = {np.mean(df[c].to_list())} ± {np.std(df[c].to_list())}\n"
    s += f'pdf size = {pdf_size/1024/1024} MB\n'
    s += f'figures size = {fig_size/1024/1024} MB\n'
    print(s.encode('utf-8'))
