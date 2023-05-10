import argparse
import os
from pathlib import Path
import sys

import numpy as np
from skimage import metrics
import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from cuda import get_memory_usage
from dataset import ReconSliceDataset, ReconVolumeDataset
from logger import Logger
from models import AFT as Model


def setup_save_dir(args) -> Path:
    save_dir = Path(args.save_dir)
    if args.resume:
        assert save_dir.is_dir()
    else:
        save_dir.mkdir(exist_ok=True)
    return save_dir


def setup_logger(resume: bool, save_dir: Path) -> None:
    if not resume and (save_dir / 'train.log').is_file():
        (save_dir / 'train.log').unlink()
    sys.stdout = Logger(sys.stdout, save_dir / 'train.log')


def setup_dataloader(args) -> tuple[DataLoader, DataLoader]:
    trainloader = torch.utils.data.DataLoader(
        ReconSliceDataset(args.data_csv, mode='multicoil_train'),
        batch_size=4, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(
        ReconVolumeDataset(args.data_csv, mode='multicoil_val', max_len=23),
        batch_size=None, shuffle=False, num_workers=1)
    return trainloader, valloader


def setup_model(save_dir: Path, args) -> Module:
    model = Model()
    if args.resume:
        model.load_state_dict(torch.load(save_dir / 'weights.pt'))
    model.to(args.device)
    return model


def setup_optimizer(model: Module):
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 1e-3},])
    return optimizer


def setup_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=1 / 10**.5, patience=2, verbose=True)
    return scheduler


def setup_loss_fun(args):
    loss_func = getattr(torch.nn.functional, args.loss_func)
    return loss_func


class TqdmExtraFormat(tqdm):
    """
    Provides a `total_time` format parameter
    Ref: https://github.com/tqdm/tqdm#description-and-additional-stats
    """
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time))
        return d


def train_epoch(model: Module, trainloader: DataLoader, loss_func, optimizer, args):
    model.train()
    all_loss = list()
    pbar = TqdmExtraFormat(
        total=14 * 60, dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {elapsed}<{remaining}/{total_time} [{rate_fmt}{postfix}]')
    for idx, (kykx, kyx, yx) in enumerate(trainloader):
        kykx = kykx.to(args.device)
        kyx = kyx.to(args.device)
        yx = yx.to(args.device)
        pred_kyx, pred_yx = model(kykx)

        loss = loss_func(torch.view_as_real(pred_kyx), torch.view_as_real(kyx)) + \
            loss_func(torch.view_as_real(pred_yx), torch.view_as_real(yx))

        optimizer.zero_grad()
        loss.backward()
        all_loss.append(loss.item())
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        pbar.n = min(int(pbar.format_dict['elapsed']), pbar.total)
        pbar.set_postfix(samples=(idx + 1) * trainloader.batch_size)
        pbar.refresh()

        if args.debug and idx == 1:
            break

        if pbar.format_dict['elapsed'] > 14 * 60:  # training should be less than 15 mins
            break
    pbar.close()
    return all_loss


def val_epoch(model: Module, valloader: DataLoader, args):
    model.eval()
    pred_ssim = list()
    with tqdm(total=len(valloader), dynamic_ncols=True) as pbar, torch.no_grad():
        for kykx, kyx, yx in valloader:
            kykx = kykx.to(args.device)
            kyx = kyx.to(args.device)
            yx = yx.to(args.device)
            pred_kyx, pred_yx = model(kykx)

            img = yx.abs().mean(dim=1).detach().cpu().numpy()
            pred_img = pred_yx.abs().mean(dim=1).detach().cpu().numpy()

            img_max = img.max()

            img /= img_max
            pred_img /= img_max

            pred_ssim.append(
                metrics.structural_similarity(img, pred_img, data_range=1)
            )

            pbar.update()
    return pred_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', required=True, help='data info csv')
    parser.add_argument('--save_dir', required=True, help='saving directory, create if it does not exist')
    parser.add_argument('--loss_func', default='mse_loss', choices=['mse_loss', 'l1_loss'])
    parser.add_argument('--device', default='cpu', help='the device on which a torch.Tensor is or will be allocated')
    parser.add_argument('--resume', action='store_true', help='continue')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    save_dir = setup_save_dir(args)
    setup_logger(args.resume, save_dir)
    trainloader, valloader = setup_dataloader(args)
    model = setup_model(save_dir, args)
    optimizer = setup_optimizer(model)
    scheduler = setup_scheduler(optimizer)
    loss_func = setup_loss_fun(args)

    best_metric = 0
    for epoch in range(1, 1000):
        all_loss = train_epoch(model, trainloader, loss_func, optimizer, args)
        pred_ssim = val_epoch(model, valloader, args)

        if np.mean(pred_ssim) > np.mean(best_metric):
            torch.save(model.state_dict(), save_dir / 'weights.pt')
            best_metric = pred_ssim

        s = f'Epoch {epoch:05d}:\n'
        s += f'    loss = {np.mean(all_loss)} ± {np.std(all_loss)}\n'
        s += f'    pid = {os.getpid()}\n'
        s += f'    GPU_memory_usage = {get_memory_usage()}\n'
        s += f'    pred_ssim = {np.mean(pred_ssim)} ± {np.std(pred_ssim)}\n'
        s += f'    best_ssim = {np.mean(best_metric)} ± {np.std(best_metric)}\n'
        print(s.encode('utf-8'))

        scheduler.step(np.mean(pred_ssim))
        if optimizer.param_groups[0]['lr'] < 1e-6:
            break

        if args.debug:
            break
