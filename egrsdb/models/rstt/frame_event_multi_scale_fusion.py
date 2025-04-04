"""Real-time Spatial Temporal Transformer.
"""
import functools
from os.path import abspath, dirname, join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from egrsdb.models.rstt.layers import (
    DecoderLayer,
    Downsample,
    EncoderLayer,
    InputProj,
    ResidualBlock_noBN,
    Upsample,
    make_layer,
)


def get_femse_small():
    model = FrameEventsMultiScaleEncoder(
        embed_dim=96,
        depths=[4, 4, 4, 4, 4, 4, 4, 4],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
        back_RBs=0,
    )
    return model


def get_femse(config):
    model = FrameEventsMultiScaleEncoder(
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        window_sizes=config.window_sizes,
        back_RBs=config.back_RBs,
    )
    return model


class EventHead(nn.Module):
    def __init__(self, moments, time_block, embed_dim):
        super(EventHead, self).__init__()
        self.time_block = time_block
        self.moments = moments
        self.embed_dim = embed_dim
        channels = time_block * embed_dim
        self.increase_dim = nn.Sequence(
            nn.Conv2d(moments, channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

    def forward(self, e):
        # e: B, M, H, W
        B, M, H, W = e.size()
        e = self.increase_dim(e)
        # B, NC, H, W -> B, N, C, H, W
        e = e.view(B, self.time_block, self.embed_dim, H, W)
        pass


class FrameEventsMultiScaleEncoder(nn.Module):
    def __init__(
        self,
        in_chans=3,
        moments=128,
        embed_dim=96,
        down_sample_type="conv",
        num_in_frames=4,
        f_depths=[8, 8, 8, 8],
        f_num_heads=[2, 4, 8, 16],
        f_window_sizes=[(4, 4), (4, 4), (4, 4), (4, 4)],
        e_depths=[8, 8, 8, 8],
        e_num_heads=[2, 4, 8, 16],
        e_window_sizes=[(4, 4), (4, 4), (4, 4), (4, 4)],
        d_depths=[8, 8, 8, 8],
        d_num_heads=[16, 8, 4, 2],
        d_window_sizes=[(4, 4), (4, 4), (4, 4), (4, 4)],
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        back_RBs=0,
    ):
        """
        Args:
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            embed_dim (int, optional): Number of projection output channels. Defaults to 32.
            depths (list[int], optional): Depths of each Transformer stage. Defaults to [2, 2, 2, 2, 2, 2, 2, 2].
            num_heads (list[int], optional): Number of attention head of each stage. Defaults to [2, 4, 8, 16, 16, 8, 4, 2].
            num_frames (int, optional): Number of input frames. Defaults to 4.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4..
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            patch_norm (bool, optional): If True, add normalization after patch embedding. Defaults to True.
            back_RBs (int, optional): Number of residual blocks for super resolution. Defaults to 10.
        """
        super().__init__()
        #
        self.num_enc_layers = len(f_depths)
        self.num_dec_layers = len(d_depths)
        self.scale = 2 ** (self.num_enc_layers - 1)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_in_frames = num_in_frames

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(f_depths[:]))]
        dec_dpr = enc_dpr[::-1]

        self.input_proj = InputProj(
            in_channels=in_chans, embed_dim=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU
        )
        self.event_proj = EventHead(moments=moments, time_block=num_in_frames - 1, embed_dim=embed_dim)

        # Frame Encoder
        self.frames_encoder_layers = nn.ModuleList()
        self.frames_downsample = nn.ModuleList()
        for i in range(self.num_enc_layers):
            frames_encoder_layer = EncoderLayer(
                dim=embed_dim,
                depth=f_depths[i],
                num_heads=f_num_heads[i],
                num_frames=num_in_frames,
                window_size=f_window_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=enc_dpr[sum(f_depths[:i]) : sum(f_depths[: i + 1])],
                norm_layer=norm_layer,
            )
            frames_downsample = Downsample(embed_dim, embed_dim, type=down_sample_type)
            self.frames_encoder_layers.append(frames_encoder_layer)
            if i != self.num_enc_layers - 1:
                self.frames_downsample.append(frames_downsample)

        # Event Encoder
        self.events_encoder_layers = nn.ModuleList()
        self.events_downsample = nn.ModuleList()
        for i in range(self.num_enc_layers):
            event_encoder_layer = EncoderLayer(
                dim=embed_dim,
                depth=e_depths[i],
                num_heads=e_num_heads[i],
                num_frames=num_frames - 1,
                window_size=e_window_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=enc_dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
            )
            event_downsample = Downsample(embed_dim, embed_dim, type=down_sample_type)
            self.events_encoder_layers.append(event_encoder_layer)
            if i != self.num_enc_layers - 1:
                self.events_downsample.append(event_downsample)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(self.num_dec_layers):
            decoder_layer = DecoderLayer(
                dim=embed_dim,
                depth=depths[i + self.num_enc_layers],
                num_heads=num_heads[i + self.num_enc_layers],
                num_frames=num_frames,
                window_size=window_sizes[i + self.num_enc_layers],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dec_dpr[sum(dec_depths[:i]) : sum(dec_depths[: i + 1])],
                norm_layer=norm_layer,
            )
            self.decoder_layers.append(decoder_layer)
            if i != self.num_dec_layers - 1:
                upsample = Upsample(embed_dim, embed_dim)
                self.upsample.append(upsample)

        # Reconstruction block
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, e):
        """
        x: frames, B, D, 3, H, W
        e: events, B, M, H, W
        """
        B, D, C, H, W = x.size()  # D input video frames
        x = self.input_proj(x)  # B, D, C, H, W
        e = self.event_proj(e)  # B, D, C, H, W

        Hp = int(np.ceil(H / self.scale)) * self.scale
        Wp = int(np.ceil(W / self.scale)) * self.scale
        x = F.pad(x, (0, Wp - W, 0, Hp - H))
        e = F.pad(e, (0, Wp - W, 0, Hp - H))

        # Encoding
        xs = []
        es = []
        for i in range(self.num_enc_layers):
            x = self.encoder_layers[i](x)
            e = self.event_encoder_layers[i](e)
            xs.append(x)
            es.append(e)
            if i != self.num_enc_layers - 1:
                x = self.downsample[i](x)
                e = self.event_downsample[i](e)
        # This code is equal as follows:
        # x1 = self.encoder_layers[0](x)
        # x1 = self.downsample[0](x1)
        # x2 = self.encoder_layers[1](x1)
        # x2 = self.downsample[1](x2)
        # x3 = self.encoder_layers[2](x2)
        # x3 = self.downsample[3](x3)
        # x4 = self.encoder_layers[3](x3)

        # e1 = self.event_encoder_layers[0](e)
        # e1 = self.event_downsample[0](e1)
        # e2 = self.event_encoder_layers[1](e1)
        # e2 = self.event_downsample[1](e2)
        # e3 = self.event_encoder_layers[2](e2)
        # e3 = self.event_downsample[2](e3)
        # e4 = self.event_encoder_layers[3](e3)

        # Fusion. x: B, D, C, H, W; e: B, D - 1, C, H, W -> y: B, 2 * D - 1, C, H, W
        y = torch.cat([x, e], dim=1)
        # y =
        # Decoding
        for i in range(self.num_dec_layers):
            y = self.decoder_layers[i](y, x_encoder_features[-i - 1])
            if i != self.num_dec_layers - 1:
                y = self.upsample[i](y)

        y = y[:, :, :, :H, :W].contiguous()
        # Super-resolution
        B, D, C, H, W = y.size()
        y = y.view(B * D, C, H, W)
        out = self.recon_trunk(y)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        _, _, H, W = out.size()
        outs = out.view(B, self.num_out_frames, -1, H, W)
        outs = outs + upsample_x.permute(0, 2, 1, 3, 4)
        return outs
