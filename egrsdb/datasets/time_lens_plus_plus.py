#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import glob
import logging
import random
from logging import info
from os import listdir
from os.path import dirname, join

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from egrsdb.datasets.basic_batch import VFI_SR_BATCH as B
from egrsdb.datasets.basic_batch import get_vfi_rs_batch
from egrsdb.utils.events_time_pyramid import event_stream_to_temporal_pyramid_representation

logging.getLogger("PIL").setLevel(logging.WARNING)

DEBUG = True
__testdata__ = join(dirname(__file__), "testdata", "time_lens_plus_plus_test")

VISUALIZE_TWO_INTERFREATURE = False
LARGE_SCALE_INTERPOLATION = 10

"""
Time Lens++ dataset structure:

├── 1_TEST
│   ├── video_name
│   │   ├── events
│   │   │   ├── 0.npy
            ...
            N.npy
│   │   ├── images
│   │   │   ├── 0.png
            ...
            N+1.png
            timestamps.txt
                ts_0
                ...
                ts_n+1
├── 2_VALIDATION
├── 3_TRAINING
"""


# def get_WHT_fixed_timestamp_coords(t: float, h: int, w: int):
#     """
#     :param t: The timestamp of the sampled frame.
#     :param h: The height of the sampled frame, which is not the original height.
#     :param w: The width of the sampled frame, which is not the original width.
#     :return: a coords tensor of shape (1, h, w, 3), where the last dimension
#              is (t, w, h).
#     """
#     assert t >= -1 and t <= 1, f"Time t should be in [-1, 1], but got {t}."
#     # (1, h, w, 3):
#     #   1 means one time stamps.
#     #   h and w means the height and width of the image.
#     #   3 means the w, h, and t coordinates. The order is important.
#     grid_map = torch.zeros(1, h, w, 3) + t
#     h_coords = torch.linspace(-1, 1, h)
#     w_coords = torch.linspace(-1, 1, w)
#     mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
#     # The feature is H W T, so the coords order is (t, w, h)
#     # grid_map \in R^{1, h, w, 3}, grid_map[:, :, :, i] is (t, w, h)
#     # grid_map[:, :, :, 1:] = torch.stack((mesh_w, mesh_h), 2)
#     grid_map[:, :, :, 1:] = torch.stack((mesh_h, mesh_w), 2)
#     return grid_map.float()


def get_timelen_pp(config):
    is_timelens = config.is_timelens
    assert not is_timelens, "This is timelens++"
    high_frame_rate = 240
    low_frame_rate = high_frame_rate / (config.skip_frames + 1)
    high_resolution = config.random_crop_resolution if config.is_random_cropping else (625, 970)

    timelens_pp_root_train = join(config.timelens_pp_root, "3_TRAINING")
    timelens_pp_root_val = join(config.timelens_pp_root, "1_TEST")

    train = TimeLensPPDataset(
        is_timelens=False,
        is_test_and_visualize=False,
        only_select_middle_frame=config.only_select_middle_frame,
        has_etpr_representation=config.has_etpr_representation,
        timelens_pp_root=timelens_pp_root_train,
        key_frames=config.key_frames,
        skip_frames=config.skip_frames,
        step=config.train_step,
        moments=config.moments,
        low_frame_rate=low_frame_rate,
        low_resolution=config.low_resolution,
        high_frame_rate=high_frame_rate,
        high_resolution=high_resolution,
        is_random_cropping=config.is_random_cropping,
        random_crop_resolution=config.random_crop_resolution,
        is_random_selection=config.is_random_selection,
        random_selection_count=config.random_selection_count,
        # pyramid
        deta_t_normal=config.deta_t_normal,
        pyramid_level=config.pyramid_level,
        pyramid_moments=config.pyramid_moments,
        pyramid_reduction_factor=config.pyramid_reduction_factor,
    )

    val = TimeLensPPDataset(
        is_timelens=False,
        is_test_and_visualize=config.is_test_and_visualize,
        only_select_middle_frame=config.only_select_middle_frame,
        has_etpr_representation=config.has_etpr_representation,
        timelens_pp_root=timelens_pp_root_val,
        key_frames=config.key_frames,
        skip_frames=config.skip_frames,
        step=config.test_step,
        moments=config.moments,
        low_frame_rate=low_frame_rate,
        low_resolution=config.low_resolution,
        high_frame_rate=high_frame_rate,
        high_resolution=high_resolution,
        is_random_cropping=config.is_random_cropping,
        random_crop_resolution=config.random_crop_resolution,
        is_random_selection=config.is_random_selection,
        random_selection_count=config.random_selection_count,
        #
        deta_t_normal=config.deta_t_normal,
        pyramid_level=config.pyramid_level,
        pyramid_moments=config.pyramid_moments,
        pyramid_reduction_factor=config.pyramid_reduction_factor,
    )
    return train, val


def get_timelen(config):
    is_timelens = config.is_timelens
    assert is_timelens, "This is timelens"
    high_frame_rate = 240
    low_frame_rate = high_frame_rate / (config.skip_frames + 1)

    test_all = join(config.timelens_test_root, "test_all")

    val = TimeLensPPDataset(
        is_timelens=True,
        is_test_and_visualize=config.is_test_and_visualize,
        only_select_middle_frame=config.only_select_middle_frame,
        has_etpr_representation=config.has_etpr_representation,
        timelens_pp_root=test_all,
        key_frames=config.key_frames,
        skip_frames=config.skip_frames,
        step=config.test_step,
        moments=config.moments,
        low_frame_rate=low_frame_rate,
        low_resolution=config.low_resolution,
        high_frame_rate=high_frame_rate,
        high_resolution=config.random_crop_resolution,
        is_random_cropping=config.is_random_cropping,
        random_crop_resolution=config.random_crop_resolution,
        is_random_selection=config.is_random_selection,
        random_selection_count=config.random_selection_count,
        #
        deta_t_normal=config.deta_t_normal,
        pyramid_level=config.pyramid_level,
        pyramid_moments=config.pyramid_moments,
        pyramid_reduction_factor=config.pyramid_reduction_factor,
    )
    return val, val


def _load_events(is_timelens, file):
    tmp = np.load(file, allow_pickle=True, mmap_mode="r")
    if is_timelens:
        (x, y, timestamp, polarity) = (
            tmp["x"].astype(np.float32).reshape((-1,)),
            tmp["y"].astype(np.float32).reshape((-1,)),
            tmp["t"].astype(np.float32).reshape((-1,)),
            tmp["p"].astype(np.float32).reshape((-1,)) * 2 - 1,
        )
    else:
        (x, y, timestamp, polarity) = (
            tmp["x"].astype(np.float32).reshape((-1,)) / 32.0,
            tmp["y"].astype(np.float32).reshape((-1,)) / 32.0,
            tmp["timestamp"].astype(np.float32).reshape((-1,)),
            tmp["polarity"].astype(np.float32).reshape((-1,)) * 2 - 1,
        )
    events = np.stack((timestamp, x, y, polarity), axis=-1)
    return events


def walk_video(is_timelens, video_path, key_frame, skip_frames, step=1):
    if is_timelens:
        image_template = join(video_path, "images_corrected", "*.png")
        event_template = join(video_path, "events_aligned", "*.npz")
        timestamps_path = join(video_path, "images_corrected", "timestamp.txt")
    else:
        image_template = join(video_path, "images", "*.png")
        event_template = join(video_path, "events", "*.npz")
        timestamps_path = join(video_path, "images", "timestamp.txt")

    images = sorted(glob.glob(image_template))
    events = sorted(glob.glob(event_template))

    timestamps = np.loadtxt(timestamps_path).tolist()
    if len(images) != (len(events) + 1):
        info(f"Warning: {video_path} has different number of images and events.")
        info(f"  -images: {len(images)}")
        info(f"  -events: {len(events)}")
        min_length = min(len(images), len(events))
        images = images[:min_length]
        events = events[: min_length - 1]

    assert len(timestamps) >= len(images)
    # Generate a sample.
    all_frames = (key_frame - 1) * (skip_frames + 1) + 1
    sample = []
    for i in range(0, len(images) - all_frames, step):
        item = {}
        item["begin_timestamp"] = timestamps[i]
        item["end_timestamp"] = timestamps[i + all_frames - 1]
        item["timestamps"] = []
        item["input_frames"] = []
        item["input_frames_timestamp"] = []
        item["gt_frames"] = []
        item["gt_frames_timestamp"] = []
        for j in range(i, i + all_frames):
            item["timestamps"].append(timestamps[j])
            if (j - i) % (skip_frames + 1) == 0:
                item["input_frames"].append(images[j])
                item["input_frames_timestamp"].append(timestamps[j])
            item["gt_frames"].append(images[j])
            item["gt_frames_timestamp"].append(timestamps[j])
        item["events"] = []
        for j in range(i, i + all_frames - 1):
            item["events"].append(events[j])
        sample.append(item)
    return sample


def walk(is_timelens, timelens_pp_root, key_frame, skip_frames, step=1):
    videos = listdir(timelens_pp_root)
    sample = []
    for video in videos:
        if video == "horse_05" or video == "horse_15" or video == "horse_16" or video == "horse_10":
            # These videos are empty.
            continue
        video_path = join(timelens_pp_root, video)
        sample.extend(walk_video(is_timelens, video_path, key_frame, skip_frames, step))
    return sample


class TimeLensPPDataset(Dataset):
    def __init__(
        self,
        is_timelens,
        is_test_and_visualize,
        only_select_middle_frame,
        has_etpr_representation,
        timelens_pp_root,
        key_frames,
        skip_frames,
        step,
        moments,
        low_frame_rate,
        low_resolution,
        high_frame_rate,
        high_resolution,
        is_random_cropping,
        random_crop_resolution,
        is_random_selection,
        random_selection_count,
        deta_t_normal,
        pyramid_level,
        pyramid_moments,
        pyramid_reduction_factor,
    ):
        super(TimeLensPPDataset, self).__init__()
        assert moments >= 2, "The moments should be greater than 2."
        # if not is_random_selection:
        #     assert random_selection_count == (key_frames - 1) * (skip_frames + 1) + 1
        if only_select_middle_frame:
            # assert key_frames == 4
            # assert skip_frames == 7
            # assert random_selection_count == 9
            # TL 的数据集是只插帧不超分，所以不用预测输入帧。
            assert random_selection_count == skip_frames

        info(f"Loading TimeLensPPDataset from {timelens_pp_root} ...")
        self.is_timelens = is_timelens
        self.is_test_and_visualize = is_test_and_visualize
        self.only_select_middle_frame = only_select_middle_frame
        self.has_etpr_representation = has_etpr_representation
        self.timelens_pp_root = timelens_pp_root
        self.key_frames = key_frames
        self.skip_frames = skip_frames * LARGE_SCALE_INTERPOLATION
        self.all_frame_count = key_frames + (key_frames - 1) * skip_frames
        self.step = step
        self.samples = walk(is_timelens, timelens_pp_root, key_frames, skip_frames, step)
        # Set the transforms.
        self.moments = moments
        self.positive = 1
        self.negative = -1
        self.is_random_cropping = is_random_cropping
        if is_random_cropping:
            self.random_crop_resolution = random_crop_resolution
        self.low_frame_rate = low_frame_rate
        self.low_resolution = low_resolution
        self.high_frame_rate = high_frame_rate
        self.high_resolution = high_resolution
        # Function
        self.to_tensor = transforms.ToTensor()
        self._info()
        # Training methods
        self.is_random_selection = is_random_selection
        self.random_selection_count = random_selection_count * LARGE_SCALE_INTERPOLATION
        self.deta_t_normal = deta_t_normal
        # random crop, the resolution is the output resolution.
        if self.is_timelens:
            self.original_resolution = None  # the resolution of TimeLens is not fixed.
        else:
            self.original_resolution = (625, 970)
        # Pyramid
        self.pyramid_level = pyramid_level
        self.pyramid_moments = pyramid_moments
        self.pyramid_reduction_factor = pyramid_reduction_factor

    def _extent_gt_timestamps(self, item):
        timestamps = item["gt_frames_timestamp"]
        gt_frames = item["gt_frames"]
        new_timestamps = []
        new_gt_frames = []
        for i in range(len(timestamps) - 1):
            for j in range(LARGE_SCALE_INTERPOLATION):
                new_timestamps.append(
                    timestamps[i] + (timestamps[i + 1] - timestamps[i]) * j / LARGE_SCALE_INTERPOLATION
                )
                new_gt_frames.append(gt_frames[i])
        new_timestamps.append(timestamps[-1])
        new_gt_frames.append(gt_frames[-1])
        item["gt_frames_timestamp"] = new_timestamps
        item["gt_frames"] = new_gt_frames
        return item

    def __getitem__(self, index):
        item = self.samples[index]

        if LARGE_SCALE_INTERPOLATION > 1:
            item = self._extent_gt_timestamps(item)

        # get video name and first frame name.
        # e.g. root/3_TRAINING/basket_04/images/000007.png
        video_name = item["input_frames"][0].split("/")[-3]
        frame_names = item["input_frames"][0].split("/")[-1].split(".")[0]
        inputs_frames_timestamps = item["input_frames_timestamp"]
        gt_frames_timestamps = item["gt_frames_timestamp"]
        # make input frames and gt frames as tensors.
        input_frames = [self.to_tensor(Image.open(i).convert("RGB")) for i in item["input_frames"]]

        if VISUALIZE_TWO_INTERFREATURE:
            input_frames[0] = input_frames[1]
            input_frames[3] = input_frames[2]

        input_frames = torch.stack(input_frames, dim=0)
        gt_frames = [self.to_tensor(Image.open(i).convert("RGB")) for i in item["gt_frames"]]
        gt_frames = torch.stack(gt_frames, dim=0)
        #
        self.original_resolution = input_frames.shape[-2:]
        # Load event data
        events_stream = [_load_events(self.is_timelens, e) for e in item["events"]]
        events_stream = np.concatenate(events_stream, axis=0)
        events_stream = self._select_events_in_resolution(events_stream)
        # make global events voxel grid with moments.
        event_voxel_grids = self._event_stream_to_voxel_grid(events_stream, self.moments, self.original_resolution)
        event_voxel_grids = event_voxel_grids.astype(np.float32)

        # Timestamps to coords
        begin_timestamp = item["begin_timestamp"]
        end_timestamp = item["end_timestamp"]
        duration = end_timestamp - begin_timestamp
        # normalize the input frames timestamps.
        inputs_frames_normal_timestamps = np.zeros(len(inputs_frames_timestamps))
        for i, input_timestamp in enumerate(inputs_frames_timestamps):
            inputs_frames_normal_timestamps[i] = (input_timestamp - begin_timestamp) / duration
        # normalize the gt frames timestamps.
        gt_frame_normal_timestamps = np.zeros((len(gt_frames_timestamps)))
        for i, gt_timestamp in enumerate(gt_frames_timestamps):
            gt_frame_normal_timestamps[i] = (gt_timestamp - begin_timestamp) / duration

        # Random select
        gt_frames, selected_indexes = self._interpolation_index_selection(gt_frames)
        # Update the gt_frames_timestamps
        gt_frame_normal_timestamps = gt_frame_normal_timestamps[selected_indexes]

        print(f"gt_frame_normal_timestamps: {len(gt_frame_normal_timestamps)}")
        print(f"selected_indexes: {selected_indexes}")

        if self.has_etpr_representation:
            # make events temporal pyramid representations. N, PL, PM, H, W
            events_temporal_pyramid_representations = self._generate_events_temporal_pyramid_representations(
                events_stream,
                gt_frame_normal_timestamps,
                self.deta_t_normal,
            )
            # to torch.floar

            events_temporal_pyramid_representations = torch.from_numpy(events_temporal_pyramid_representations).float()
        else:
            events_temporal_pyramid_representations = "None(str)"

        inputs_frames_normal_timestamps = torch.FloatTensor(inputs_frames_normal_timestamps)
        gt_frame_normal_timestamps = torch.FloatTensor(gt_frame_normal_timestamps)
        # Data augmentation
        # random crop
        if self.is_random_cropping:
            input_frames, gt_frames, event_voxel_grids, events_temporal_pyramid_representations = self.random_crop(
                input_frames, gt_frames, event_voxel_grids, events_temporal_pyramid_representations
            )
        event_voxel_grids = torch.from_numpy(event_voxel_grids)
        # down-sample input frames
        lr_h, lr_w = self.low_resolution
        rc_h, rc_w = self.random_crop_resolution
        if (lr_h != rc_h) or (lr_w != rc_w):
            info(f"Down-sample input frames from {self.random_crop_resolution} to {self.low_resolution}")
            N, C, H, W = input_frames.shape
            down_sample_size = self.low_resolution
            input_frames = F.interpolate(input_frames, size=down_sample_size, mode="bilinear", align_corners=False)
            event_voxel_grids = event_voxel_grids.unsqueeze(1)
            event_voxel_grids = F.interpolate(
                event_voxel_grids, size=down_sample_size, mode="bilinear", align_corners=False
            )
            event_voxel_grids = event_voxel_grids.squeeze(1)
            #
            if self.has_etpr_representation:
                N, PL, PM, H, W = events_temporal_pyramid_representations.shape
                events_temporal_pyramid_representations = rearrange(
                    events_temporal_pyramid_representations, "n pl pm h w -> (n pl pm) 1 h w"
                )
                events_temporal_pyramid_representations = F.interpolate(
                    events_temporal_pyramid_representations, size=down_sample_size, mode="bilinear", align_corners=False
                )
                events_temporal_pyramid_representations = rearrange(
                    events_temporal_pyramid_representations, "(n pl pm) 1 h w -> n pl pm h w", n=N, pl=PL, pm=PM
                )

        # make a batch
        batch = get_vfi_rs_batch()
        batch[B.RANDOM_SELECTION] = self.is_random_selection
        #
        batch[B.VIDEO_NAME] = video_name
        batch[B.FRAME_NAMES] = frame_names
        batch[B.INPUT_FRAME_COUNT] = self.key_frames
        batch[B.OUTPUT_FRAME_COUNT] = self.all_frame_count
        batch[B.SKIP_FRAME_COUNT] = self.skip_frames
        #
        batch[B.LOW_FRAME_RATE] = self.low_frame_rate
        batch[B.LOW_RESOLUTION] = self.low_resolution
        batch[B.HIGH_FRAME_RATE] = self.high_frame_rate
        batch[B.HIGH_RESOLUTION] = self.high_resolution

        batch[B.LFR_LR_FRAMES_TIMESTAMPS_RAW] = inputs_frames_timestamps
        batch[B.LFR_LR_FRAMES_TIMESTAMPS_NORMAL] = inputs_frames_normal_timestamps
        batch[B.HFR_HR_FRAMES_TIMESTAMPS_RAW] = gt_frames_timestamps
        batch[B.HFR_HR_FRAMES_TIMESTAMPS_NORMAL] = gt_frame_normal_timestamps

        batch[B.LFR_LR_FRAMES] = input_frames
        batch[B.EVENTS_GLOBAL_MOMENTS] = event_voxel_grids
        batch[B.HFR_HR_FRAMES] = gt_frames
        batch[B.EVENTS_PYRAMID_REPRESENTATION_MOMENTS] = events_temporal_pyramid_representations
        return batch

    def __len__(self):
        return len(self.samples)

    def _select_events_in_resolution(self, events):
        H, W = self.original_resolution
        if events.shape[0] == 0:
            return events
        # Remove the events outside the image
        x_max = np.max(events[:, 1])
        x_min = np.min(events[:, 1])
        y_max = np.max(events[:, 2])
        y_min = np.min(events[:, 2])
        if x_max >= W or x_min < 0:
            events = events[events[:, 1] < W]
        if y_max >= H or y_min < 0:
            events = events[events[:, 2] < H]
        return events

    def _interpolation_index_selection(self, gt_tensor):
        N, C, H, W = gt_tensor.shape

        if self.only_select_middle_frame:
            # The input frames are 4.
            # The skip frames are 7.
            # [(0),1,2,3,4,5,6,7,(8),9,10,11,12,13,14,15,(16),17,18,19,20,21,22,23,(24)]
            # the_middle_index = [8, 9, 10, 11, 12, 13, 14, 15, 16]
            # TL 的数据集是只插帧不超分，所以不用预测输入帧。
            one_step = self.skip_frames + 1
            the_middle_index = [i for i in range(one_step + 1, 2 * one_step, 1)]
            new_gt_tensor = torch.zeros((self.random_selection_count, C, H, W))
            for i, index in enumerate(the_middle_index):
                new_gt_tensor[i] = gt_tensor[index]
            return new_gt_tensor, np.array(the_middle_index)

        if self.is_random_selection:
            selected_indexes = sorted(random.sample(range(N), self.random_selection_count))
            new_gt_tensor = torch.zeros((self.random_selection_count, C, H, W))
            for i, index in enumerate(selected_indexes):
                new_gt_tensor[i] = gt_tensor[index]
            return new_gt_tensor, np.array(selected_indexes)
        else:  # no random selection
            return gt_tensor, np.array(list(range(N)))

    def _event_stream_to_voxel_grid(self, events_stream, moments, resolution):
        """
        :param events_stream: The events stream.
        :param moments: The moments.
        :param resolution: The resolution of the voxel grid.
        :param positive: The positive value.
        :param negative: The negative value.
        :return: A list of voxel grid images. [M, H, W]
        """
        # The voxel grid is a list of i
        voxel_grid = np.zeros((moments, resolution[0], resolution[1]), dtype=np.float32)

        # The voxel grid is a 3D image.
        start_time = events_stream[:, 0].min()
        end_time = events_stream[:, 0].max()
        voxel_grid_time_step = (end_time - start_time) / moments
        for i in range(moments):
            left_time = start_time + i * voxel_grid_time_step
            right_time = start_time + (i + 1) * voxel_grid_time_step
            # The voxel grid in a moment is a 2D image.
            left_index = np.searchsorted(events_stream[:, 0], left_time, side="left")
            right_index = np.searchsorted(events_stream[:, 0], right_time, side="right")
            li, ri = left_index, right_index
            x, y, p = events_stream[li:ri, 1], events_stream[li:ri, 2], events_stream[li:ri, 3]
            x = x.astype(np.int32)
            y = y.astype(np.int32)
            voxel_grid[i] = self._render(x=x, y=y, p=p, shape=resolution)
        return voxel_grid

    def _generate_events_temporal_pyramid_representations(
        self,
        events_stream,
        selected_norm_timestamp,
        deta_t,
    ):
        start_time = events_stream[:, 0].min()
        end_time = events_stream[:, 0].max()
        during_time = end_time - start_time

        N, PL, PM = self.random_selection_count, self.pyramid_level, self.pyramid_moments
        H, W = self.original_resolution
        event_temporal_pryamid_representations = np.zeros((N, PL, PM, H, W))
        for i, normal_timestamp in enumerate(selected_norm_timestamp):
            left_t = (normal_timestamp - deta_t) * during_time + start_time
            right_t = (normal_timestamp + deta_t) * during_time + start_time
            # selected_local_events = events_stream[events_stream[:, 0] >= left_t]
            # selected_local_events = selected_local_events[selected_local_events[:, 0] < right_t]
            left_index = np.searchsorted(events_stream[:, 0], left_t, side="left")
            right_index = np.searchsorted(events_stream[:, 0], right_t, side="right")
            selected_local_events = events_stream[left_index:right_index]
            # events, pyramid_level, pyramid_moments, reduction_factor, resolution
            event_temporal_pryamid_representations[i] = event_stream_to_temporal_pyramid_representation(
                events=selected_local_events,
                pyramid_level=self.pyramid_level,
                pyramid_moments=self.pyramid_moments,
                reduction_factor=self.pyramid_reduction_factor,
                resolution=self.original_resolution,
            )
        return event_temporal_pryamid_representations

    @staticmethod
    def _render(x, y, p, shape):
        # info(f"render: x:max:{x.max()}, min:{x.min()}, y:max:{y.max()}, min:{y.min()}, p:max:{p.max()}, min:{p.min()}")
        events = np.zeros(shape=shape, dtype=np.float32)
        events[y, x] = p
        return events

    def _info(self):
        info(f"  -key_frame= {self.key_frames}")
        info(f"  -skip_frame= {self.skip_frames}")
        info(f"    -in = {self.key_frames}")
        info(f"    -out= {self.all_frame_count}")
        info(f"  -step= {self.step}")
        info(f"  -moments = {self.moments}")
        info(f"  -random_cropping= {self.is_random_cropping}")
        info(f"  -num samples: {len(self.samples)}")

    def random_crop(self, input_frames, gt_frames, events, events_temporal_pyramid_representations):
        random_crop_h, random_crop_w = self.random_crop_resolution
        H, W = self.original_resolution
        if self.is_test_and_visualize:
            x1, y1 = 0, 0
        else:
            x1 = random.randint(0, H - random_crop_h)
            y1 = random.randint(0, W - random_crop_w)
        x2 = x1 + random_crop_h
        y2 = y1 + random_crop_w
        input_frames = input_frames[:, :, x1:x2, y1:y2]
        gt_frames = gt_frames[:, :, x1:x2, y1:y2]
        events = events[:, x1:x2, y1:y2]
        if self.has_etpr_representation:
            events_temporal_pyramid_representations = events_temporal_pyramid_representations[:, :, :, x1:x2, y1:y2]
        return input_frames, gt_frames, events, events_temporal_pyramid_representations
