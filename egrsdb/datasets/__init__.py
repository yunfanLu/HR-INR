#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from egrsdb.datasets.adobe240fps_sr_vfi_dataset import get_adobe240fps_sr_vfi_dataset
from egrsdb.datasets.color_event_dataset import get_ced_dataset
from egrsdb.datasets.davis_rs_event import get_dre_dataset
from egrsdb.datasets.demosaic_hybridevs_dataset import (
    get_demosaic_hybridevs_dataset,
    get_demosaic_hybridevs_val_dataset,
)
from egrsdb.datasets.evunroll_real_dataset import get_evunroll_real_dataset
from egrsdb.datasets.evunroll_simulated_dataset import get_evunroll_simulated_dataset_with_config
from egrsdb.datasets.fastec_rsb import get_fastec_rolling_shutter_blur_dataset
from egrsdb.datasets.gev_rsb import get_gev_rolling_shutter_blur_dataset
from egrsdb.datasets.time_lens_plus_plus import get_timelen, get_timelen_pp
from egrsdb.datasets.aplx_rgbe_isp_dataset import get_alpx_rgbe_isp_dataset


def get_dataset(config):
    if config.NAME == "get_alpx_rgbe_isp_dataset":
        return get_alpx_rgbe_isp_dataset(
            alpx_rgbe_isp_root=config.alpx_rgbe_isp_root,
            moments=config.moments,
            in_frame=config.in_frame,
            future_frame=config.future_frame,
            past_frame=config.past_frame,
            using_events=config.using_events,
            using_interpolate=config.using_interpolate,
            random_crop_resolution=config.random_crop_resolution,
            evaluation_visualization=config.evaluation_visualization,
        )
    elif config.NAME == "demosaic-hybridevs":
        return get_demosaic_hybridevs_dataset(config)
    elif config.NAME == "demosaic-hybridevs-val":
        return get_demosaic_hybridevs_val_dataset(config)
    elif config.NAME == "gev-rolling-shutter-blur":
        return get_gev_rolling_shutter_blur_dataset(
            root=config.root,
            blur_accumulate=config.blur_accumulate,
            gs_sharp_frame_count=config.gs_sharp_frame_count,
            events_moment=config.events_moment,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            is_color=config.is_color,
            gs_sharp_start_index=config.gs_sharp_start_index,
            gs_sharp_end_index=config.gs_sharp_end_index,
            calculate_in_linear_domain=config.calculate_in_linear_domain,
            event_for_gs_frame_buffer=config.event_for_gs_frame_buffer,
            correct_offset=config.correct_offset,
        )
    elif config.NAME == "fastec-rolling-shutter-blur":
        return get_fastec_rolling_shutter_blur_dataset(
            root=config.root,
            blur_accumulate=config.blur_accumulate,
            gs_sharp_frame_count=config.gs_sharp_frame_count,
            events_moment=config.events_moment,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            is_color=config.is_color,
            gs_sharp_start_index=config.gs_sharp_start_index,
            gs_sharp_end_index=config.gs_sharp_end_index,
            calculate_in_linear_domain=config.calculate_in_linear_domain,
            event_for_gs_frame_buffer=config.event_for_gs_frame_buffer,
            correct_offset=config.correct_offset,
        )
    elif config.NAME == "evunroll-real":
        test = get_evunroll_real_dataset(
            fps=20.79,
            data_root=config.data_root,
            moments=config.events_moment,
            is_color=config.is_color,
        )
        return test, test
    elif config.NAME == "evunroll-simulated":
        return get_evunroll_simulated_dataset_with_config(config)
    # vfi + sr
    elif config.NAME == "timelens++_vfi_sr":
        return get_timelen_pp(config)
    elif config.NAME == "timelens_vfi_sr":
        return get_timelen(config)
    elif config.NAME == "adobe240fps_stsr":
        return get_adobe240fps_sr_vfi_dataset(config)
    elif config.NAME == "ced-sr":
        return get_ced_dataset(config)
    elif config.NAME == "dre-real":
        return get_dre_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {config.NAME}")
