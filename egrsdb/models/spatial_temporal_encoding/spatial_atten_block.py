import pudb
import torch.nn as nn
import torch.nn.functional as F
from absl.logging import info

from egrsdb.models.egvsr.ltb import MLABlock
from egrsdb.models.egvsr.patch_options import extract_patches, restore_patches


class SpatialAttentBlock(nn.Module):
    def __init__(
        self, is_reslution_downsample, feature_h_w, in_channels, reduce_channels, patch_size, dim, drop=0.0, depth=1
    ):
        super(SpatialAttentBlock, self).__init__()

        self.is_reslution_downsample = is_reslution_downsample
        if is_reslution_downsample:
            self.feature_h_w = feature_h_w[0] // 2, feature_h_w[1] // 2
        else:
            self.feature_h_w = feature_h_w

        self.reduce_channels = reduce_channels
        self.patch_size = patch_size
        self.dim = dim
        pzh, pzw = patch_size

        self.reduce_unfold_channel_mlp = nn.Linear(reduce_channels * pzh * pzw, dim)
        self.mlab = MLABlock(dim, drop, depth)
        self.restore_unfold_channel_mlp = nn.Linear(dim, reduce_channels * pzh * pzw)

        if is_reslution_downsample:
            self.reduce_channel_conv = nn.Sequential(
                nn.Conv2d(in_channels, reduce_channels, 1),
                nn.ReLU(),
                nn.Conv2d(reduce_channels, reduce_channels, 3, stride=2, padding=1),
            )
            self.restore_channel_conv = nn.Sequential(
                nn.Conv2d(reduce_channels, reduce_channels, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_channels, in_channels, 3, stride=2, padding=1, output_padding=1),
            )
        else:
            self.reduce_channel_conv = nn.Sequential(nn.Conv2d(in_channels, reduce_channels, 1), nn.ReLU())
            self.restore_channel_conv = nn.Sequential(nn.Conv2d(reduce_channels, in_channels, 1), nn.ReLU())

        info(f"Init SpatialAttentBlock")
        info(f"  in_channels    : {in_channels}")
        info(f"  reduce_channels: {reduce_channels}")
        info(f"  patch_size     : {patch_size}")
        info(f"  dim            : {dim}")
        info(f"  drop           : {drop}")
        info(f"  depth          : {depth}")

    def forward(self, x):
        # 1. Downsample Channel Dimion
        y = self.reduce_channel_conv(x)
        # 2. Extract Patches
        pzh, pzw = self.patch_size
        y = extract_patches(y, kernel_sizes=self.patch_size, strides=[1, 1], rates=[1, 1])
        y = y.permute(0, 2, 1)
        y = F.relu(self.reduce_unfold_channel_mlp(y))
        y = self.mlab(y)
        y = F.relu(self.restore_unfold_channel_mlp(y))
        y = y.permute(0, 2, 1)
        y = restore_patches(y, out_size=self.feature_h_w, ksizes=pzh, strides=1, padding=pzh // 2)
        # 3. Upsample Channel Dimion
        y = self.restore_channel_conv(y)
        return y + x
