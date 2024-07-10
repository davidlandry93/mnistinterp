import torch
import torch.nn as nn

from .blocks import UNet


class UNetModel(nn.Module):
    def __init__(
        self, in_channels=1, embedding_size=64, ch_mults=(1, 2, 4, 8), out_channels=1
    ):
        super().__init__()

        self.unet = UNet(
            in_dim=in_channels + 2,
            embed_dim=embedding_size,
            out_dim=out_channels,
            dim_scales=ch_mults,
        )

    def forward(self, xt, t):
        batch_size, width, height = xt.shape[0], xt.shape[2], xt.shape[3]

        if len(t.shape) == 0:
            t = t.unsqueeze(0).repeat(batch_size)

        horizontal = torch.linspace(0.0, 1.0, width, device=xt.device)
        vertical = torch.linspace(0.0, 1.0, height, device=xt.device)
        xx, yy = torch.meshgrid(horizontal, vertical, indexing="ij")

        x = torch.concat(
            [
                xt,
                xx.reshape(1, 1, width, height).repeat(batch_size, 1, 1, 1),
                yy.reshape(1, 1, width, height).repeat(batch_size, 1, 1, 1),
            ],
            dim=1,
        )

        return self.unet(x, t)
