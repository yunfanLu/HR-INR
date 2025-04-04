#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/12/24 15:57
from enum import Enum, unique


def get_rs_deblur_inference_batch():
    return {
        "video_name": "NONE",
        "rolling_blur_frame_name": "NONE",
        # input: events
        "events": "NONE",
        "events_for_gs_sharp_frames": "NONE",
        # input: rolling blur frame
        # color image only for reference
        "rolling_blur_frame_color": "NONE",
        # input gray image
        "rolling_blur_frame_gray": "NONE",
        "rolling_blur_start_time": "NONE",
        "rolling_blur_end_time": "NONE",
        "rolling_blur_exposure_time": "NONE",
        # output: rolling blur frame
        "rolling_sharp_pred_frames": "NONE",
        "rolling_blur_pred_frame": "NONE",
        # input: global sharp frame
        "global_sharp_frame_timestamps": "NONE",
        "global_sharp_frames": "NONE",
        # Output: global sharp frame
        "global_sharp_pred_frames": "NONE",
        "global_sharp_pred_frames_differential": "NONE",  # List[] N x B
    }


@unique
class DemosaicHybridevsBatch(Enum):
    IMAGE_NAME = "image_name"
    HEIGHT = "height"
    WIDTH = "width"
    RAW_TENSOR = "raw_image"
    GROUND_TRUTH = "ground_truth"
    RAW_RGB_POSITION = "raw_rgb_position"
    PREDICTION = "prediction"


def get_demosaic_batch():
    batch = {}
    for item in DemosaicHybridevsBatch:
        batch[item] = "NONE(str)"
    return batch


@unique
class EventRAWISPBatch(Enum):
    INPUT_FRAME_COUNT = "input_frame_count"
    VIDEO_NAME = "video_name"
    FRAME_NAME = "frame_name"
    HEIGHT = "height"
    WIDTH = "width"
    RAW_TENSORS = "raw_tensors"
    RAW_TIMESTAMPS = "raw_timestamps"

    GROUND_TRUTH = "ground_truth"
    GROUND_TRUTH_TIMESTAMPS = "ground_truth_timestamps"
    PREDICTION = "prediction"

    EVENTS_VOXEL_GRID = "events_voxel_grid"
    EVENTS_VOXEL_GRID_TIMESTAMPS_START = "events_voxel_grid_timestamps_start"
    EVENTS_VOXEL_GRID_TIMESTAMPS_END = "events_voxel_grid_timestamps_end"


def get_rgbe_isp_batch():
    batch = {}
    for item in EventRAWISPBatch:
        batch[item] = "NONE(str)"
    return batch


# VFI + SR batch data structure.
# Do not consider rolling shutter or global shutter. All inputs consider as global shutter.
@unique
class VFI_SR_BATCH(Enum):
    # training pattern
    #   if random selection is used, each inference batch will only predict one frame.
    RANDOM_SELECTION = "random_selection"
    # input file names to identify the video
    VIDEO_NAME = "video_name"
    FRAME_NAMES = "frame_names"
    INPUT_FRAME_COUNT = "input_frame_count"
    OUTPUT_FRAME_COUNT = "output_frame_count"
    SKIP_FRAME_COUNT = "skip_frame_count"
    # input: events
    EVENTS = "events"  # the event stream
    EVENTS_GLOBAL_MOMENTS = "events_global_moments"  # the event moments, moments x H x W x 2
    EVENTS_PYRAMID_REPRESENTATION_MOMENTS = (
        "EVENTS_PYRAMID_REPRESENTATION_MOMENTS"  # the event moments, [moments x H x W x 2] x HFR_TIMESTAMPS
    )
    # input: low frame-rate and low-resolution frames
    LFR_LR_FRAMES = "lfr_lr_frames"
    LOW_FRAME_RATE = "low_frame_rate"
    LOW_RESOLUTION = "low_resolution"
    LFR_LR_FRAMES_TIMESTAMPS_RAW = "lfr_lr_frames_timestamps_raw"
    LFR_LR_FRAMES_TIMESTAMPS_NORMAL = "lfr_lr_frames_timestamps_normal"
    # ground truth: high frame-rate and high-resolution frames
    HFR_HR_FRAMES = "hfr_hr_frames"
    HIGH_FRAME_RATE = "high_frame_rate"
    HIGH_RESOLUTION = "high_resolution"
    HFR_HR_FRAMES_TIMESTAMPS_RAW = "hfr_hr_frames_timestamps_raw"
    HFR_HR_FRAMES_TIMESTAMPS_NORMAL = "hfr_hr_frames_timestamps_normal"
    HFR_HR_FRAMES_TIMESTAMPS_COORDS = "hfr_hr_frames_timestamps_coords"
    # output: high frame-rate and high-resolution frames
    HFR_HR_FRAMES_PRED = "hfr_hr_frames_pred"
    # Middle Feature
    TIME_EMBEDDED_FEATURES = "time_embedded_features"
    # Visulization of middle feature
    REGIONAL_EVENT_FEATURES = "regional_event_features"
    HOLISTIC_EVENT_FRAME_FEATURES = "holistic_event_features"


def get_vfi_rs_batch():
    batch = {}
    for item in VFI_SR_BATCH:
        batch[item] = "NONE(str)"
    return batch
