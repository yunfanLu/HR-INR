from typing import Tuple

import torch
import torch.nn.functional as F
from absl.logging import error, info, warning
from torch import nn

from egrsdb.datasets.basic_batch import VFI_SR_BATCH as B
from egrsdb.functions.rolling_coords import get_t_global_shutter_coordinate
from egrsdb.models.els_net.esl_backbone import ESLBackBone
from egrsdb.models.rstt.event_pyramid_swin_transformer_encder import EventPyramidRepresentationSwinTransformerEncoder
from egrsdb.models.rstt.frame_event_multi_scale_fusion_v2 import get_standard_femse
from egrsdb.models.rstt.RSTT_frame_event_encoder import RSTTwithEventAdapter
from egrsdb.models.spatial_temporal_encoding.event_pyramid_encoding import EventPyramidEncoding3xConv
from egrsdb.models.spatial_temporal_encoding.frame_event_encoding import FramesEventSpatialTemporalEncoding
from egrsdb.models.temporal_spatial_embedding.blinear_spatial_embedding import (
    VFISRConv1x1DecoderLearnedPositionEmbedding,
)
from egrsdb.models.temporal_spatial_embedding.temporal_spatial_embedding import TemporalSpatialEmbedding


class ESLConfigDefine:
    hidden_channels = 128
    high_dim_channels = 128
    is_deformable = False
    loop = 25
    has_scn_loop = False


class SpatialTemporalSuperResolution(nn.Module):
    def __init__(
        self,
        image_channel: int,
        input_frames: int,
        output_frames: int,
        events_global_moments: int,
        encoder_name: str,
        encoder_config: dict,
        inr_temporal_in_channel: int,
        inr_temporal_out_channel: int,
        global_inr_and_etpr_fusion_type: str,
        decoder_name: str,
        decoder_config: dict,
        epr_encoder_name: str,
        epr_encoder_config: dict,
        low_resolution: Tuple[int, int],
        low_frame_rate: int,
        gt_resolution: Tuple[int, int],
        high_frame_rate: int,
        intermediate_visualization: bool,
    ):
        """CVPR 2024 Main Model"""
        super(SpatialTemporalSuperResolution, self).__init__()
        # assert
        assert image_channel in [3], "image_channel should be 3, CVPR 24 paper not support gray image."
        assert input_frames >= 2, "input_frames should be positive"
        assert global_inr_and_etpr_fusion_type in [
            "add",
            "concat",
            "none",
        ], "global_inr_and_etpr_fusion_type should be add or concat"
        # 1. set attributes
        # 1.0 input setting
        self.image_channel = image_channel
        self.spatial_coords_dim = 2
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.events_global_moments = events_global_moments
        # 1.1 network setting
        self.encoder_name = encoder_name
        self.encoder_config = encoder_config
        self.inr_temporal_in_channel = inr_temporal_in_channel
        self.inr_temporal_out_channel = inr_temporal_out_channel
        self.global_inr_and_etpr_fusion_type = global_inr_and_etpr_fusion_type
        self.decoder_name = decoder_name
        self.decoder_config = decoder_config
        self.epr_encoder_name = epr_encoder_name
        self.epr_encoder_config = epr_encoder_config
        # 1.2 data setting
        self.low_resolution = low_resolution
        self.low_frame_rate = low_frame_rate
        self.gt_resolution = gt_resolution
        self.high_frame_rate = high_frame_rate
        # 1.3 visualization setting
        self.intermediate_visualization = intermediate_visualization

        # 2. build network
        # 2.1 encoder
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        #
        self.pyramid_representation_events_encoder = self._get_pyramid_events_encoder()
        # 3. info
        self._info()

    def forward(self, batch):
        # 1. load data from batch
        # 1.1 inputs
        global_events = batch[B.EVENTS_GLOBAL_MOMENTS]
        pyramid_representation_events = batch[B.EVENTS_PYRAMID_REPRESENTATION_MOMENTS]
        if len(global_events.shape) == 5:
            # this is event two channel frames, it need to  be reshape to one channel
            batch_size, moments, C_e, H, W = global_events.shape
            global_events = global_events.reshape(batch_size, moments * C_e, H, W)

        input_frames = batch[B.LFR_LR_FRAMES]
        batch_size, N_in, C, H, W = input_frames.shape
        # Reshape for inference
        # B, N_in, C, H, W -> B, N_in*C, H, W
        # input_frames = input_frames.reshape(batch_size, N_in * C, H, W)

        # 1.2 inference config
        low_resolution = batch[B.LOW_RESOLUTION]
        high_resolution = batch[B.HIGH_RESOLUTION]
        hr_h, hr_w = high_resolution
        gt_frames_timestamps = batch[B.HFR_HR_FRAMES_TIMESTAMPS_NORMAL]
        # 2. encoder
        global_inr = self.encoder(global_events, input_frames)
        # 3. decoder
        batch_size, N_out = gt_frames_timestamps.shape
        output_frames = []
        t_embedded_features = []
        regional_event_features_visualization = []
        for i in range(N_out):
            # gt_frames_timestamps: [B, Timestamp_In]
            gt_frame_ts = gt_frames_timestamps[:, i]
            # query_coords = get_t_global_shutter_coordinate(t=gt_frame_ts, h=hr_h, w=hr_w, with_position=True)
            out_resolution = batch[B.HIGH_RESOLUTION]
            # for the random scale upsampling the output of batch is different. So when the batch size > 1, we need to
            # check the out_resolution is the same.
            # warning(f"out_resolution: {out_resolution}")
            if batch_size > 1:
                h_list = out_resolution[0]
                w_list = out_resolution[1]
                oh, ow = h_list[0], w_list[0]
                for b in range(1, batch_size):
                    boh, bow = h_list[b], w_list[b]
                    if boh != oh or bow != ow:
                        error(f"batch size > 1, but out_resolution is not the same")
                out_resolution = [oh, ow]

            if self.epr_encoder_name != "none":
                # B, N_out, PL, PM, H, W
                event_pyramid_local = pyramid_representation_events[:, i, :, :, :, :]
                event_pyramid_local_inr = self.pyramid_representation_events_encoder(event_pyramid_local)
                if self.intermediate_visualization:
                    regional_event_features_visualization.append(event_pyramid_local_inr.mean(dim=1))

                if self.global_inr_and_etpr_fusion_type == "add":
                    global_inr_with_local_information = global_inr + event_pyramid_local_inr
                elif self.global_inr_and_etpr_fusion_type == "concat":
                    global_inr_with_local_information = torch.cat([global_inr, event_pyramid_local_inr], dim=1)
                else:
                    raise NotImplementedError
            else:
                global_inr_with_local_information = global_inr

            sH, sW = out_resolution
            # the input of decoder is : inr, t, sH, sW
            output_frame, t_embedded_feature = self.decoder(
                inr=global_inr_with_local_information, t=gt_frame_ts, sH=sH, sW=sW
            )
            output_frames.append(output_frame)
            t_embedded_features.append(t_embedded_feature)

        output_frames = torch.stack(output_frames, dim=1)
        t_embedded_features = torch.stack(t_embedded_features, dim=1)
        # 4. store data to batch
        if self.intermediate_visualization:
            batch[B.REGIONAL_EVENT_FEATURES] = torch.stack(regional_event_features_visualization, dim=1)
            batch[B.HOLISTIC_EVENT_FRAME_FEATURES] = global_inr.mean(dim=1)
        batch[B.HFR_HR_FRAMES_PRED] = output_frames
        batch[B.TIME_EMBEDDED_FEATURES] = t_embedded_features
        return batch

    def _info(self):
        info(f"{__class__}")
        info(f"image_channel: {self.image_channel}")
        info(f"input_frames: {self.input_frames}")
        info(f"output_frames: {self.output_frames}")
        info(f"events_global_moments: {self.events_global_moments}")
        info(f"encoder_name: {self.encoder_name}")
        info(f"decoder_name: {self.decoder_name}")
        info(f"low_resolution: {self.low_resolution}")
        info(f"low_frame_rate: {self.low_frame_rate}")
        info(f"gt_resolution: {self.gt_resolution}")
        info(f"high_frame_rate: {self.high_frame_rate}")
        info(f"intermediate_visualization: {self.intermediate_visualization}")

    # Encoder: the input is the global event and two rgb frame.
    # The output is the global inr feature.
    def _get_encoder(self):
        if self.encoder_name == "esl_backbone":
            esl_config = ESLConfigDefine()
            esl_config.high_dim_channels = self.inr_temporal_in_channel
            #
            is_color = self.image_channel == 3
            return ESLBackBone(
                input_frames=self.input_frames,
                is_color=is_color,
                event_moments=self.events_global_moments,
                hidden_channels=esl_config.hidden_channels,
                high_dim_channels=esl_config.high_dim_channels,
                is_deformable=esl_config.is_deformable,
                loop=esl_config.loop,
                has_scn_loop=esl_config.has_scn_loop,
            )
        elif self.encoder_name == "sptial_temporal_transformer":
            return FramesEventSpatialTemporalEncoding(
                spatial_temporal_attention_loop=self.encoder_config.spatial_temporal_attention_loop,
                frame_resolution=self.encoder_config.frame_resolution,
                frame_channels=3,
                event_channels=self.encoder_config.event_channels,
                in_frames=2,
                transformer_resolution_downsample=self.encoder_config.transformer_resolution_downsample,
                feature_channels=self.encoder_config.feature_channels,
                transformer_patch_size=self.encoder_config.transformer_patch_size,
                transformer_dim=self.encoder_config.transformer_dim,
                transformer_reduce_channels=self.encoder_config.transformer_reduce_channels,
                transformer_drop=self.encoder_config.transformer_drop,
                transformer_depth=self.encoder_config.transformer_depth,
                temporal_encoding_depth=self.encoder_config.temporal_encoding_depth,
                temporal_encoding_loop=self.encoder_config.temporal_encoding_loop,
                temporal_encoding_cal_kernel_size=self.encoder_config.temporal_encoding_cal_kernel_size,
            )
        elif self.encoder_name == "rstt":
            return RSTTwithEventAdapter(
                only_train_adapter=self.encoder_config.only_train_adapter,
                rstt_type=self.encoder_config.rstt_type,
                input_frame=4,
                events_moments=self.events_global_moments,
                rstt_pretrain_type=self.encoder_config.rstt_pretrain_type,
                event_adapter_type="SCN",
                event_adatper_config=self.encoder_config.event_adatper_config,
            )
        elif self.encoder_name == "femse_v2":
            return get_standard_femse(self.encoder_config)
        else:
            raise NotImplementedError

    # Decoder: the input is the global inr feature and the query coord.
    # The output is the rgb frame.
    def _get_decoder(self):
        if self.decoder_name == "conv1x1_learn_position_encoding":
            model = VFISRConv1x1DecoderLearnedPositionEmbedding(
                coords_dim=3,
                global_inr_channel=self.inr_temporal_in_channel,
                local_inr_channel=self.inr_temporal_in_channel,
                hidden_channels=self.decoder_config.decoder_hidden_channel,
                out_channels=self.image_channel,
            )
        elif self.decoder_name == "large_temporal_spatial_embedding":
            model = TemporalSpatialEmbedding(
                inr_temporal_in_channel=self.inr_temporal_in_channel,
                inr_temporal_out_channel=self.inr_temporal_out_channel,
                temporal_embedding_type=self.decoder_config.temporal_embedding_type,
                spatial_embedding_type=self.decoder_config.spatial_embedding_type,
                spatial_embedding_config=self.decoder_config.spatial_embedding_config,
            )
        else:
            raise ValueError(f"Unknown {self.decoder_name}")
        return model

    def _get_pyramid_events_encoder(self):
        if self.global_inr_and_etpr_fusion_type == "add":
            assert self.inr_temporal_in_channel == self.epr_encoder_config.epr_out_channel
        elif self.global_inr_and_etpr_fusion_type == "concat":
            # 672 == 7 * 96 is the swin inr out channel
            etpr_channel = self.epr_encoder_config.epr_out_channel
            assert self.inr_temporal_in_channel == etpr_channel + self.encoder_config.final_inr_dim
        elif self.global_inr_and_etpr_fusion_type == "none":
            assert self.inr_temporal_in_channel == self.encoder_config.final_inr_dim

        if self.epr_encoder_name == "3xconv":
            return EventPyramidEncoding3xConv(
                pyramid_level=self.epr_encoder_config.pyramid_level,
                pyramid_moments=self.epr_encoder_config.pyramid_moments,
                epre_channel=self.epr_encoder_config.epre_channel,
                epr_out_channel=self.epr_encoder_config.epr_out_channel,
            )
        elif self.epr_encoder_name == "swin":
            return EventPyramidRepresentationSwinTransformerEncoder(
                pyramid_level=self.epr_encoder_config.pyramid_level,
                pyramid_moments=self.epr_encoder_config.pyramid_moments,
                epre_channel=self.epr_encoder_config.epre_channel,
                epr_out_channel=self.epr_encoder_config.epr_out_channel,
            )
        return None
