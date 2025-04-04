import torch
from torch import nn
from torch.nn import functional as F

from egrsdb.models.spatial_temporal_encoding.spatial_atten_block import SpatialAttentBlock
from egrsdb.models.spatial_temporal_encoding.temporal_atten_block import TemporalAttentionBlock


class FramesEventSpatialTemporalEncoding(nn.Module):
    def __init__(
        self,
        spatial_temporal_attention_loop,
        frame_resolution,
        frame_channels,
        event_channels,
        in_frames,
        feature_channels,
        transformer_resolution_downsample,
        transformer_reduce_channels,
        transformer_patch_size,
        transformer_dim,
        transformer_drop,
        transformer_depth,
        temporal_encoding_depth,
        temporal_encoding_loop,
        temporal_encoding_cal_kernel_size,
    ):
        super(FramesEventSpatialTemporalEncoding, self).__init__()
        # check input args

        self.spatial_temporal_attention_loop = spatial_temporal_attention_loop
        self.frame_resolution = frame_resolution
        self.H, self.W = frame_resolution
        self.frame_channels = frame_channels
        self.event_channels = event_channels
        self.in_frames = in_frames
        self.feature_channels = feature_channels
        self.transformer_resolution_downsample = transformer_resolution_downsample
        self.transformer_patch_size = transformer_patch_size
        self.tpz_h, self.tpz_w = transformer_patch_size
        self.transformer_dim = transformer_dim
        self.transformer_drop = transformer_drop
        self.transformer_depth = transformer_depth
        self.transformer_reduce_channels = transformer_reduce_channels
        self.temporal_encoding_depth = temporal_encoding_depth
        self.temporal_encoding_loop = temporal_encoding_loop
        self.temporal_encoding_cal_kernel_size = temporal_encoding_cal_kernel_size

        # frame and event head
        self.frame_head = nn.Conv2d(frame_channels * in_frames, feature_channels, 3, padding=1)
        self.event_head = nn.Sequential(
            nn.Conv2d(event_channels, feature_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
        )
        # Spatial temporal encoding
        self.spatial_temporal_attention = nn.Sequential()
        for i in range(spatial_temporal_attention_loop):
            self.spatial_temporal_attention.append(self._make_spatial_temporal_atten_block())

    def forward(self, event, frame):
        if len(frame.shape) == 5:
            b, f, c, h, w = frame.shape
            frame = frame.reshape(b, f * c, h, w)
        if len(event.shape) == 5:
            b, f, c, h, w = event.shape
            event = event.reshape(b, f * c, h, w)
        # Head
        frame_feature = F.relu(self.frame_head(frame))
        event_feature = F.relu(self.event_head(event))
        spatial_temporal_feature = frame_feature * event_feature
        # Spatial temporal encoding
        spatial_temporal_feature = self.spatial_temporal_attention(spatial_temporal_feature)
        return spatial_temporal_feature

    def _make_spatial_temporal_atten_block(self):
        temproal_encoding = TemporalAttentionBlock(
            in_channels=self.feature_channels,
            depth=self.temporal_encoding_depth,
            loop_count=self.temporal_encoding_loop,
            cal_kernel_size=self.temporal_encoding_cal_kernel_size,
        )
        spatial_encoding = SpatialAttentBlock(
            is_reslution_downsample=self.transformer_resolution_downsample,
            feature_h_w=(self.H, self.W),
            in_channels=self.feature_channels,
            reduce_channels=self.transformer_reduce_channels,
            patch_size=(self.tpz_h, self.tpz_w),
            dim=self.transformer_dim,
            drop=self.transformer_drop,
            depth=self.transformer_depth,
        )
        spatial_temporal_attention = nn.Sequential(temproal_encoding, spatial_encoding)
        return spatial_temporal_attention
