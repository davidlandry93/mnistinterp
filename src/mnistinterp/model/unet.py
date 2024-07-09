import torch
import torch.nn as nn

from .blocks import UNet


class UNetModel(nn.Module):
    def __init__(self, in_channels=1, n_channels=64, ch_mults=(1, 2, 4, 8)):
        super().__init__()

        self.unet = UNet(
            in_dim=in_channels + 2,
            embed_dim=n_channels,
            out_dim=1,
            dim_scales=ch_mults,
        )

    def forward(self, batch):
        x, t = batch["xt"], batch["t"]

        batch_size, width, height = x.shape[0], x.shape[2], x.shape[3]

        horizontal = torch.linspace(0.0, 1.0, width, device=x.device)
        vertical = torch.linspace(0.0, 1.0, height, device=x.device)
        xx, yy = torch.meshgrid(horizontal, vertical, indexing="ij")

        x = torch.concat(
            [
                x,
                xx.reshape(1, 1, width, height).repeat(batch_size, 1, 1, 1),
                yy.reshape(1, 1, width, height).repeat(batch_size, 1, 1, 1),
            ],
            dim=1,
        )

        return self.unet(x, t)
