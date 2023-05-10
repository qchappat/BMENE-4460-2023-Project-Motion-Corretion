import argparse
import os
from pathlib import Path
import shutil
import sys
import json

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
from dataset import AccVolumeDataset
from logger import Logger
from models import AFT_ResCUNet as Model
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


def load_json(file_path: str):
    """Load from a JSON file.

    Parameters
    ----------
    file_path : str
        File path of the JSON file to load.

    Returns
    -------
    data : dict
        Loaded data from the JSON file.
    """

    with open(file_path) as json_file:
        data = json.load(json_file)

    return data


def setup_dataloader(args) -> DataLoader:
    corresp_dict = load_json(args.json_path)
    testloader = torch.utils.data.DataLoader(
        AccVolumeDataset(args.data_csv, mode='multicoil_test', corresp_dict=corresp_dict, max_len=None),
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
        for kspace, ispace, original_kspace, original_ispace in testloader:
            pred_ispace = torch.cat([model(slc.unsqueeze(0).to(args.device)) for slc in kspace])

            img = original_ispace.abs().mean(dim=1).numpy()
            corrupted_img = ispace.abs().mean(dim=1).numpy()
            predicted_img = pred_ispace.abs().mean(dim=1).cpu().numpy()

            img /= img.max()
            corrupted_img /= corrupted_img.max()
            predicted_img /= predicted_img.max()

            input_metrics = dict()
            pred_metrics = dict()
            input_metrics['SSIM'] = metrics.structural_similarity(img, corrupted_img, data_range=1)
            pred_metrics['SSIM'] = metrics.structural_similarity(img, predicted_img, data_range=1)
            input_metrics['PSNR'] = metrics.peak_signal_noise_ratio(img, corrupted_img, data_range=1)
            pred_metrics['PSNR'] = metrics.peak_signal_noise_ratio(img, predicted_img, data_range=1)
            input_metrics['NRMSE'] = metrics.normalized_root_mse(img, corrupted_img)
            pred_metrics['NRMSE'] = metrics.normalized_root_mse(img, predicted_img)

            input_df = pd.DataFrame.from_dict(input_metrics, orient='index').T
            pred_df = pd.DataFrame.from_dict(pred_metrics, orient='index').T
            all_df.append(pd.concat([input_df, pred_df], axis=1, keys=['Input', 'Prediction']))

            fig = plot(img, corrupted_img, predicted_img, input_metrics, pred_metrics)
            fig.set_size_inches(14, 12)
            fig.savefig(
                save_dir / 'figures' / f"{pbar.n:04d}_{input_metrics['SSIM']:.5f}_{pred_metrics['SSIM']:.5f}_{pred_metrics['SSIM']-input_metrics['SSIM']:.5f}.png",
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
    parser.add_argument('--json_path', required=True, help='dictionnary path')
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
        s += f"Input_{c} = {np.mean(df['Input'][c].to_list())} ± {np.std(df['Input'][c].to_list())}\n"
        s += f"Prediction_{c} = {np.mean(df['Prediction'][c].to_list())} ± {np.std(df['Prediction'][c].to_list())}\n"
    s += f'pdf size = {pdf_size/1024/1024} MB\n'
    s += f'figures size = {fig_size/1024/1024} MB\n'
    print(s.encode('utf-8'))
