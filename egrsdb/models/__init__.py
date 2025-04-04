from egrsdb.models.demosaic.unet_image_swin import get_unet_image_swin
from egrsdb.models.els_net.els_net import get_elsnet
from egrsdb.models.rsdb import get_rolling_shutter_deblur_model
from egrsdb.models.stsr import SpatialTemporalSuperResolution
from egrsdb.models.pynet.pynet import PyNET
from egrsdb.models.rgbe_isp.unet_2d_rgbe_isp import RGBEISPUNet
from egrsdb.models.rgbe_isp.cameranet import CameraNet
from egrsdb.models.rgbe_isp.awnet.model import AWNet
from egrsdb.models.rgbe_isp.unet_image_swin_rgbe_isp import get_unet_image_swin_rgbe_isp
from egrsdb.models.rgbe_isp.ispenet.ispe_net import ISPESLNet
from egrsdb.models.rgbe_isp.unet_2d_events_rgbe_isp import RGBEventISPUNet


def get_model(config):
    if config.NAME == "CameraNet":
        return CameraNet()
    elif config.NAME == "ISPESLNet":
        return ISPESLNet(moments=config.moments)
    elif config.NAME == "RGBEventISPUNet":
        return RGBEventISPUNet(in_channels=config.in_channels, moments=config.moments, with_events=config.with_events)
    elif config.NAME == "AWNet":
        return AWNet()
    elif config.NAME == "unet_image_swin_rgbe_isp":
        return get_unet_image_swin_rgbe_isp(config)
    elif config.NAME == "RGBEISPUNet":
        return RGBEISPUNet()
    elif config.NAME == "pynet":
        # return PyNET(level=0, instance_norm=True, instance_norm_level_1=True)
        return PyNET(
            level=config.level, instance_norm=config.instance_norm, instance_norm_level_1=config.instance_norm_level_1
        )
    elif config.NAME == "rsdb":
        return get_rolling_shutter_deblur_model(
            image_channel=config.image_channel,
            coords_dim=config.coords_dim,
            events_moment=config.events_moment,
            meta_type=config.meta_type,
            encoder_name=config.encoder_name,
            decoder_name=config.decoder_name,
            inr_depth=config.inr_depth,
            inr_in_channel=config.inr_in_channel,
            inr_mid_channel=config.inr_mid_channel,
            image_height=config.image_height,
            image_width=config.image_width,
            rs_blur_timestamp=config.rs_blur_timestamp,
            gs_sharp_count=config.gs_sharp_count,
            rs_integral=config.rs_integral,
            intermediate_visualization=config.intermediate_visualization,
            dcn_config=config.dcn_config,
            esl_config=config.esl_config,
            correct_offset=config.correct_offset,
            time_embedding_type=config.time_embedding_type,
        )
    elif config.NAME == "els_net":
        return get_elsnet(scale=config.scale, is_color=config.is_color)
    # vfi + sr
    elif config.NAME == "stsr":
        return SpatialTemporalSuperResolution(
            image_channel=config.image_channel,
            input_frames=config.input_frames,
            output_frames=config.output_frames,
            events_global_moments=config.events_global_moments,
            encoder_name=config.encoder_name,
            encoder_config=config.encoder_config,
            inr_temporal_in_channel=config.inr_temporal_in_channel,
            inr_temporal_out_channel=config.inr_temporal_out_channel,
            global_inr_and_etpr_fusion_type=config.global_inr_and_etpr_fusion_type,
            decoder_name=config.decoder_name,
            decoder_config=config.decoder_config,
            epr_encoder_name=config.epr_encoder_name,
            epr_encoder_config=config.epr_encoder_config,
            low_resolution=config.low_resolution,
            low_frame_rate=config.low_frame_rate,
            gt_resolution=config.gt_resolution,
            high_frame_rate=config.high_frame_rate,
            intermediate_visualization=config.intermediate_visualization,
        )
    elif config.NAME == "unet_image_swin":
        return get_unet_image_swin(config)
    else:
        raise ValueError(f"Model {config.NAME} is not supported.")
