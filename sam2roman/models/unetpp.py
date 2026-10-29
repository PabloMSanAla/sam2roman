"""
UNet++ (Nested U-Net) architecture for dense astronomical image segmentation.
Paper: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard double convolution block:
    [Conv2d -> Norm -> GELU -> Conv2d -> Norm -> GELU]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
    ):
        super().__init__()
        if norm_type == "group":
            groups = min(8, out_channels)
            norm1 = nn.GroupNorm(groups, out_channels)
            norm2 = nn.GroupNorm(groups, out_channels)
        else:
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm1,
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm2,
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """
    Nested U-Net (UNet++) architecture for astronomical source segmentation.

    Connects the encoder and decoder sub-networks through a series of nested,
    dense skip pathways, reducing the semantic gap between encoder and decoder
    feature maps and improving boundary delineation for astronomical sources.

    Parameters
    ----------
    in_channels : int, default 3
        Number of input channels (3 for leakySB RGB, 1 for raw float32 FITS).
    num_classes : int, default 3
        Number of segmentation classes (0: Sky background, 1: Source interior, 2: Boundary/edge).
    deep_supervision : bool, default False
        If True, returns predictions from all intermediate nodes X^{0,1}, X^{0,2}, X^{0,3}, X^{0,4}.
    features : tuple of int, default (32, 64, 128, 256, 512)
        Number of feature channels at each depth level.
    norm_type : str, default 'batch'
        Normalization layer type ('batch' or 'group').
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        deep_supervision: bool = False,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        norm_type: str = "batch",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        nb = list(features)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Node level 0 (Encoder backbone)
        self.conv0_0 = ConvBlock(in_channels, nb[0], norm_type=norm_type)
        self.conv1_0 = ConvBlock(nb[0], nb[1], norm_type=norm_type)
        self.conv2_0 = ConvBlock(nb[1], nb[2], norm_type=norm_type)
        self.conv3_0 = ConvBlock(nb[2], nb[3], norm_type=norm_type)
        self.conv4_0 = ConvBlock(nb[3], nb[4], norm_type=norm_type)

        # Node level 1 (Nested skip pathways)
        self.conv0_1 = ConvBlock(nb[0] + nb[1], nb[0], norm_type=norm_type)
        self.conv1_1 = ConvBlock(nb[1] + nb[2], nb[1], norm_type=norm_type)
        self.conv2_1 = ConvBlock(nb[2] + nb[3], nb[2], norm_type=norm_type)
        self.conv3_1 = ConvBlock(nb[3] + nb[4], nb[3], norm_type=norm_type)

        # Node level 2
        self.conv0_2 = ConvBlock(nb[0] * 2 + nb[1], nb[0], norm_type=norm_type)
        self.conv1_2 = ConvBlock(nb[1] * 2 + nb[2], nb[1], norm_type=norm_type)
        self.conv2_2 = ConvBlock(nb[2] * 2 + nb[3], nb[2], norm_type=norm_type)

        # Node level 3
        self.conv0_3 = ConvBlock(nb[0] * 3 + nb[1], nb[0], norm_type=norm_type)
        self.conv1_3 = ConvBlock(nb[1] * 3 + nb[2], nb[1], norm_type=norm_type)

        # Node level 4 (Final output node X^{0,4})
        self.conv0_4 = ConvBlock(nb[0] * 4 + nb[1], nb[0], norm_type=norm_type)

        # Prediction heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W).

        Returns
        -------
        torch.Tensor or List[torch.Tensor]
            Logits of shape (B, num_classes, H, W).
            If deep_supervision=True and self.training, returns list of [out1, out2, out3, out4].
        """
        # Column 0: Backbone Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Column 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            if self.training:
                return [out1, out2, out3, out4]
            else:
                return out4
        else:
            return self.final(x0_4)
