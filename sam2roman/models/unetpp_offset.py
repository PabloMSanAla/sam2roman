"""
UNet++ with Criss-Cross Attention (CCA) and Multi-Task Center Offset Heads.
===========================================================================

Combines:
1. UNet++ nested dense skip connection backbone.
2. Recurrent Criss-Cross Attention (RCCA) at bottleneck and deep decoder nodes:
   - Full global horizontal and vertical context with linear O(HW(H+W)) memory.
   - Connects distant astronomical diffraction spike pixels (200-500+ px) to central stellar cores.
3. Multi-task output heads:
   - Semantic Head: Source foreground probability logits (1 channel or multi-class).
   - Center Heatmap Head: Gaussian centroid keypoint logits (1 channel).
   - Center Offset Head: Dense 2D displacement vectors (2 channels: delta_y, delta_x).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2roman.models.unetpp import ConvBlock


class CrissCrossAttention(nn.Module):
    """
    Criss-Cross Attention (CCA) module for long-range spatial context.
    Paper: Huang et al., "CCNet: Criss-Cross Attention for Semantic Segmentation" (ICCV 2019 / TPAMI 2020).

    Computes attention affinities along the horizontal row and vertical column of each pixel,
    achieving long-range global connectivity with O(HW(H+W)) computational complexity.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    inter_channels : int, optional
        Intermediate channel dimension for Query and Key projections (defaults to in_channels // 8).
    """

    def __init__(self, in_channels: int, inter_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(16, in_channels // 8)

        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor:
            Context-enhanced tensor of shape (B, C, H, W).
        """
        b, c, h, w = x.shape

        # Query: (B, C', H, W) -> permute to (B, W, H, C') and (B, H, W, C')
        proj_query = self.query_conv(x)
        proj_query_h = proj_query.permute(0, 3, 1, 2).contiguous().view(b * w, self.inter_channels, h).permute(0, 2, 1)
        proj_query_w = proj_query.permute(0, 2, 1, 3).contiguous().view(b * h, self.inter_channels, w).permute(0, 2, 1)

        # Key: (B, C', H, W)
        proj_key = self.key_conv(x)
        proj_key_h = proj_key.permute(0, 3, 1, 2).contiguous().view(b * w, self.inter_channels, h)
        proj_key_w = proj_key.permute(0, 2, 1, 3).contiguous().view(b * h, self.inter_channels, w)

        # Value: (B, C, H, W)
        proj_value = self.value_conv(x)
        proj_value_h = proj_value.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        proj_value_w = proj_value.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)

        # Calculate energy along height and width
        # (B*W, H, H) and (B*H, W, W)
        energy_h = torch.bmm(proj_query_h, proj_key_h).view(b, w, h, h).permute(0, 2, 1, 3)
        energy_w = torch.bmm(proj_query_w, proj_key_w).view(b, h, w, w)

        # Concatenate along the cross dimension (H + W): (B, H, W, H + W)
        concate = torch.cat([energy_h, energy_w], dim=-1)
        att_score = F.softmax(concate, dim=-1)

        att_h = att_score[:, :, :, :h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        att_w = att_score[:, :, :, h:].contiguous().view(b * h, w, w)

        out_h = torch.bmm(proj_value_h, att_h.permute(0, 2, 1)).view(b, w, c, h).permute(0, 2, 3, 1)
        out_w = torch.bmm(proj_value_w, att_w.permute(0, 2, 1)).view(b, h, c, w).permute(0, 2, 1, 3)

        out = out_h + out_w
        return self.gamma * out + x


class RecurrentCrissCrossAttention(nn.Module):
    """
    Recurrent Criss-Cross Attention (RCCA).
    Two consecutive passes of Criss-Cross Attention with shared parameters
    enable every pixel to capture all-to-all global context across the entire image.
    """

    def __init__(self, in_channels: int, recurrence: int = 2):
        super().__init__()
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for _ in range(self.recurrence):
            out = self.cca(out)
        return out


class UNetPlusPlusCenterOffset(nn.Module):
    """
    Nested U-Net (UNet++) with Criss-Cross Attention and Multi-Task Center Offset Heads.

    Outputs three concurrent spatial predictions:
    1. 'semantic': Foreground segmentation logits (B, semantic_classes, H, W).
    2. 'center': Center keypoint probability logits (B, 1, H, W).
    3. 'offset': 2D displacement vectors (B, 2, H, W) pointing to (delta_y, delta_x).

    Parameters
    ----------
    in_channels : int, default 3
        Number of input channels (3 for leakySB RGB, 1 for raw float32 FITS).
    semantic_classes : int, default 1
        Semantic foreground classes (1 for binary foreground BCE/Dice, 2 for bg/fg, 3 for bg/core/edge).
    features : tuple of int, default (32, 64, 128, 256, 512)
        Channel dimensions per depth level.
    use_cca : bool, default True
        Whether to equip the bottleneck and deep decoder with Recurrent Criss-Cross Attention.
    deep_supervision : bool, default False
        Whether to predict multi-task outputs from intermediate nodes X^{0,1..4}.
    norm_type : str, default 'batch'
        Normalization layer ('batch' or 'group').
    """

    def __init__(
        self,
        in_channels: int = 3,
        semantic_classes: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        use_cca: bool = True,
        deep_supervision: bool = False,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.semantic_classes = semantic_classes
        self.features = features
        self.use_cca = use_cca
        self.deep_supervision = deep_supervision
        nb = list(features)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Node level 0 (Backbone encoder)
        self.conv0_0 = ConvBlock(in_channels, nb[0], norm_type=norm_type)
        self.conv1_0 = ConvBlock(nb[0], nb[1], norm_type=norm_type)
        self.conv2_0 = ConvBlock(nb[1], nb[2], norm_type=norm_type)
        self.conv3_0 = ConvBlock(nb[2], nb[3], norm_type=norm_type)
        self.conv4_0 = ConvBlock(nb[3], nb[4], norm_type=norm_type)

        # Optional Criss-Cross Attention modules at bottleneck and deep decoder
        if self.use_cca:
            self.cca_bottleneck = RecurrentCrissCrossAttention(nb[4], recurrence=2)
            self.cca_deep = RecurrentCrissCrossAttention(nb[3], recurrence=2)
        else:
            self.cca_bottleneck = nn.Identity()
            self.cca_deep = nn.Identity()

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

        # Node level 4 (Final nested node X^{0,4})
        self.conv0_4 = ConvBlock(nb[0] * 4 + nb[1], nb[0], norm_type=norm_type)

        # Specialized Multi-Task Heads attached to X^{0,4}
        self.head_semantic = nn.Sequential(
            nn.Conv2d(nb[0], nb[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nb[0]) if norm_type == "batch" else nn.GroupNorm(min(8, nb[0]), nb[0]),
            nn.GELU(),
            nn.Conv2d(nb[0], semantic_classes, kernel_size=1),
        )

        self.head_center = nn.Sequential(
            nn.Conv2d(nb[0], nb[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nb[0]) if norm_type == "batch" else nn.GroupNorm(min(8, nb[0]), nb[0]),
            nn.GELU(),
            nn.Conv2d(nb[0], 1, kernel_size=1),
        )

        self.head_offset = nn.Sequential(
            nn.Conv2d(nb[0], nb[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nb[0]) if norm_type == "batch" else nn.GroupNorm(min(8, nb[0]), nb[0]),
            nn.GELU(),
            nn.Conv2d(nb[0], 2, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W).

        Returns
        -------
        Dict[str, torch.Tensor]:
            - 'semantic': Logits of shape (B, semantic_classes, H, W)
            - 'center': Heatmap logits of shape (B, 1, H, W)
            - 'offset': Offset vector field of shape (B, 2, H, W)
        """
        # Column 0: Backbone Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Long-range context injection at bottleneck
        x4_0 = self.cca_bottleneck(x4_0)

        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x3_1 = self.cca_deep(x3_1)

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Column 4 (Node X^{0,4})
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # Multi-task predictions
        semantic_out = self.head_semantic(x0_4)
        center_out = self.head_center(x0_4)
        offset_out = self.head_offset(x0_4)

        return {
            "semantic": semantic_out,
            "center": center_out,
            "offset": offset_out,
        }
