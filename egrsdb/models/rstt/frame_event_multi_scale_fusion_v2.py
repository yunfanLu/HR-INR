"""Real-time Spatial Temporal Transformer.
"""
import functools
from os.path import abspath, dirname, join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from egrsdb.models.rstt.layers import DecoderLayer, Downsample, EncoderLayer, InputProj, Upsample


def get_femse_large(moments, final_inr_dim):
    model = FrameEventsMultiScaleEncoderV2(
        moments=moments,
        embed_dim=96,
        depths=[8, 8, 8, 8, 8, 8, 8, 8],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
        back_RBs=0,
        final_inr_dim=final_inr_dim,
    )
    return model


def get_femse_medium(moments, final_inr_dim):
    model = FrameEventsMultiScaleEncoderV2(
        moments=moments,
        embed_dim=96,
        depths=[6, 6, 6, 6, 6, 6, 6, 6],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
        back_RBs=0,
        final_inr_dim=final_inr_dim,
    )
    return model


def get_femse_small(moments, final_inr_dim):
    model = FrameEventsMultiScaleEncoderV2(
        moments=moments,
        embed_dim=96,
        depths=[4, 4, 4, 4, 4, 4, 4, 4],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]],
        back_RBs=0,
        final_inr_dim=final_inr_dim,
    )
    return model


def get_standard_femse(config):
    if config.size == "large":
        return get_femse_large(config.moments, final_inr_dim=config.final_inr_dim)
    elif config.size == "medium":
        return get_femse_medium(config.moments, final_inr_dim=config.final_inr_dim)
    else:
        return get_femse_small(config.moments, final_inr_dim=config.final_inr_dim)


def get_raw_femse(config):
    model = FrameEventsMultiScaleEncoderV2(
        in_chans=3,
        moments=config.moments,
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        window_sizes=config.window_sizes,
        back_RBs=config.back_RBs,
        final_inr_dim=config.final_inr_dim,
    )
    return model


class FrameEventFusion(nn.Module):
    def __init__(self, embed_dim=96, num_frames=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.mlp = nn.Conv2d(embed_dim * (2 * num_frames - 1), num_frames * embed_dim, 1, 1, 0)

    def forward(self, x, e):
        """
        Args:
            x: B, N, C, H, W
            e: B, N - 1, C, H, W
        return: B, N, C, H, W
        """
        xe = torch.cat([x, e], dim=1)
        B, N2, C, H, W = xe.size()
        xe = xe.view(B, N2 * C, H, W)
        xe = self.mlp(xe)
        B, NC, H, W = xe.size()
        xe = xe.view(B, self.num_frames, self.embed_dim, H, W)
        return xe


class EventHead(nn.Module):
    def __init__(self, moments, time_block, embed_dim):
        super(EventHead, self).__init__()
        self.time_block = time_block
        self.moments = moments
        self.embed_dim = embed_dim
        channels = time_block * embed_dim
        self.increase_dim = nn.Sequential(
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
        return e


class FrameEventsMultiScaleEncoderV2(nn.Module):
    def __init__(
        self,
        in_chans=3,
        moments=96,
        embed_dim=96,
        depths=[8, 8, 8, 8, 8, 8, 8, 8],
        num_heads=[2, 4, 8, 16, 16, 8, 4, 2],
        window_sizes=[(4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4), (4, 4)],
        num_frames=4,
        final_inr_dim=672,
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

        self.num_layers = len(depths)
        self.num_enc_layers = self.num_layers // 2
        self.num_dec_layers = self.num_layers // 2
        self.scale = 2 ** (self.num_enc_layers - 1)
        dec_depths = depths[self.num_enc_layers :]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_in_frames = num_frames
        self.num_out_frames = 2 * num_frames - 1

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[: self.num_enc_layers]))]
        dec_dpr = enc_dpr[::-1]

        self.input_proj = InputProj(
            in_channels=in_chans, embed_dim=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU
        )
        self.event_proj = EventHead(moments=moments, time_block=num_frames - 1, embed_dim=embed_dim)

        # Encoder Frame
        self.encoder_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i_layer in range(self.num_enc_layers):
            encoder_layer = EncoderLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_frames=num_frames,
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=enc_dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
            )
            downsample = Downsample(embed_dim, embed_dim)
            self.encoder_layers.append(encoder_layer)
            self.downsample.append(downsample)

        # Encoder Event
        self.e_encoder_layers = nn.ModuleList()
        self.e_downsample = nn.ModuleList()
        for i_layer in range(self.num_enc_layers):
            e_encoder_layer = EncoderLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_frames=num_frames - 1,
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=enc_dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
            )
            e_downsample = Downsample(embed_dim, embed_dim)
            self.e_encoder_layers.append(e_encoder_layer)
            self.e_downsample.append(e_downsample)

        self.frame_event_fusions = nn.ModuleList()
        for i_layer in range(self.num_dec_layers):
            frame_event_fusion = FrameEventFusion(embed_dim=embed_dim, num_frames=num_frames)
            self.frame_event_fusions.append(frame_event_fusion)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i_layer in range(self.num_dec_layers):
            decoder_layer = DecoderLayer(
                dim=embed_dim,
                depth=depths[i_layer + self.num_enc_layers],
                num_heads=num_heads[i_layer + self.num_enc_layers],
                num_frames=num_frames,
                window_size=window_sizes[i_layer + self.num_enc_layers],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dec_dpr[sum(dec_depths[:i_layer]) : sum(dec_depths[: i_layer + 1])],
                norm_layer=norm_layer,
            )
            self.decoder_layers.append(decoder_layer)
            if i_layer != self.num_dec_layers - 1:
                upsample = Upsample(embed_dim, embed_dim)
                self.upsample.append(upsample)

        # # Reconstruction block
        # ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=embed_dim)
        # self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)
        # # Upsampling
        # self.upconv1 = nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(embed_dim, 64 * 4, 3, 1, 1, bias=True)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        # self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        # self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        # # Activation function
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.final_inr_adapter = nn.Conv2d(embed_dim * (2 * num_frames - 1), final_inr_dim, 1, 1, 0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, e, x):
        B, D, C, H, W = x.size()  # D input video frames
        # x = x.permute(0, 2, 1, 3, 4)
        # upsample_x = F.interpolate(x, (2 * D - 1, H * 4, W * 4), mode="trilinear", align_corners=False)
        # x = x.permute(0, 2, 1, 3, 4)

        x = self.input_proj(x)  # B, D, C, H, W
        e = self.event_proj(e)  # B, D - 1, C, H, W

        Hp = int(np.ceil(H / self.scale)) * self.scale
        Wp = int(np.ceil(W / self.scale)) * self.scale
        x = F.pad(x, (0, Wp - W, 0, Hp - H))
        e = F.pad(e, (0, Wp - W, 0, Hp - H))

        encoder_features = []
        for i_layer in range(self.num_enc_layers):
            x = self.encoder_layers[i_layer](x)
            encoder_features.append(x)
            if i_layer != self.num_enc_layers - 1:
                x = self.downsample[i_layer](x)

        e_encoder_features = []
        for i_layer in range(self.num_enc_layers):
            e = self.e_encoder_layers[i_layer](e)
            e_encoder_features.append(e)
            if i_layer != self.num_enc_layers - 1:
                e = self.e_downsample[i_layer](e)

        # _, _, C, h, w = x.size()
        # # TODO: Use interpolation for queries
        # y = torch.zeros((B, self.num_out_frames, C, h, w), device=x.device)
        # for i in range(self.num_out_frames):
        #     if i % 2 == 0:
        #         y[:, i, :, :, :] = x[:, i // 2]
        #     else:
        #         y[:, i, :, :, :] = (x[:, i // 2] + x[:, i // 2 + 1]) / 2
        y = torch.cat([x, e], dim=1)

        for i_layer in range(self.num_dec_layers):
            x = encoder_features[-i_layer - 1]
            e = e_encoder_features[-i_layer - 1]
            ex = self.frame_event_fusions[i_layer](x, e)
            y = self.decoder_layers[i_layer](y, ex)
            if i_layer != self.num_dec_layers - 1:
                y = self.upsample[i_layer](y)

        y = y[:, :, :, :H, :W].contiguous()
        # Super-resolution
        B, D, C, H, W = y.size()
        y = y.view(B, D * C, H, W)
        # out = self.recon_trunk(y)
        # out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        # out = self.lrelu(self.HRconv(out))
        # out = self.conv_last(out)
        # _, _, H, W = out.size()
        # outs = out.view(B, self.num_out_frames, -1, H, W)
        # outs = outs + upsample_x.permute(0, 2, 1, 3, 4)
        outs = self.final_inr_adapter(y)
        return outs
