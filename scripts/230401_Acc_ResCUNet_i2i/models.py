import torch
from torch import nn, Tensor
from torch.nn import Module

from cplxtorch import nn as cnn
import cplxtorchvision
from cplxtorchvision.models import UNet


def basicblock(in_features: int, inter_features: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        cnn.Linear(in_features, inter_features),
        cnn.Linear(inter_features, out_features)
    )


class AFT(Module):
    def __init__(self, H: int = 640, W: int = 320) -> None:
        super().__init__()
        self.linear1 = basicblock(W, W, W)
        self.linear2 = basicblock(H, H, H)

    def forward(self, kykx: Tensor) -> Tensor:
        kyx = self.linear1(kykx)
        xky = torch.transpose(kyx, -2, -1)
        xy = self.linear2(xky)
        yx = torch.transpose(xy, -2, -1)
        return kyx, yx


cplxtorchvision.models.unet.norm_type = 'GroupNorm'


class ResCUNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.cunet = UNet(4, 4, [32, 64, 128, 256, 512], True)

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        t = self.cunet(input)  # featured_ispace -> residual
        output = identity + t  # featured_ispace + residual -> predicted_ispace
        return output


class AFT_ResCUNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.aft = AFT()
        self.rescunet = ResCUNet()

    def forward(self, input: Tensor) -> Tensor:
        _, t = self.aft(input)  #original_kspace -> featured_ispace
        output = self.rescunet(t)  #featured_ispace -> predicted_ispace
        return output

class ResCUNet_AFT_ResCUNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.rescunet1 = ResCUNet()
        self.aft = AFT()
        self.rescunet2 = ResCUNet()

    def forward(self, input: Tensor) -> Tensor:
        residual_1 = self.rescunet1(input)  # original_kspace -> featured_k-space
        _, residual_2 = self.aft(residual_1)  # featured_kspace -> featured_ispace
        output = self.rescunet2(residual_2)  # featured_ispace -> predicted_ispace
        return output
    
class ResCUNet_AFT(Module):
    def __init__(self) -> None:
        super().__init__()
        self.rescunet = ResCUNet()
        self.aft = AFT()

    def forward(self, input: Tensor) -> Tensor:
        t = self.rescunet(input)  # original_kspace -> featured_k-space
        _, output = self.aft(t)  # featured_kspace -> predicted_ispace
        return output