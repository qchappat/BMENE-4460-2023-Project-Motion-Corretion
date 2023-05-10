import numpy as np
import torch


def ifft(a, axis=-1, norm=None):
    a = np.fft.ifftshift(a, axes=axis)
    a = np.fft.ifft(a, axis=axis, norm=norm)
    a = np.fft.fftshift(a, axes=axis)
    return a


def ifft2(a, axes=(-2, -1), norm=None):
    a = np.fft.ifftshift(a, axes=axes)
    a = np.fft.ifft2(a, axes=axes, norm=norm)
    a = np.fft.fftshift(a, axes=axes)
    return a


def fft2(a, axes=(-2, -1), norm=None):
    a = np.fft.ifftshift(a, axes=axes)
    a = np.fft.fft2(a, axes=axes, norm=norm)
    a = np.fft.fftshift(a, axes=axes)
    return a


def ifft2_tensor(a, dim=(-2, -1), norm=None):
    a = torch.fft.ifftshift(a, dim=dim)
    a = torch.fft.ifft2(a, dim=dim, norm=norm)
    a = torch.fft.fftshift(a, dim=dim)
    return a


def fft2_tensor(a, dim=(-2, -1), norm=None):
    a = torch.fft.ifftshift(a, dim=dim)
    a = torch.fft.fft2(a, dim=dim, norm=norm)
    a = torch.fft.fftshift(a, dim=dim)
    return a


def fft_tensor(a, dim=-1, norm=None):
    a = torch.fft.ifftshift(a, dim=dim)
    a = torch.fft.fft(a, dim=dim, norm=norm)
    a = torch.fft.fftshift(a, dim=dim)
    return a
