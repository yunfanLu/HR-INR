import logging
from os import listdir
from os.path import join
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from absl.logging import info
from einops import rearrange, reduce, repeat
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from egrsdb.datasets.basic_batch import VFI_SR_BATCH as B
from egrsdb.datasets.basic_batch import get_vfi_rs_batch
from egrsdb.utils.events_time_pyramid import event_stream_to_temporal_pyramid_representation

logging.getLogger("PIL").setLevel(logging.WARNING)


DEBUG = False


def get_ced_time_stamp(ced_file):
    file_name = Path(ced_file).stem
    return float(file_name)


def get_ced_dataset(config):
    train_video, test_videos = [], []
    all_video = sorted(listdir(config.ced_root))
    # These two video have different resolution with other videos.
    all_video.remove("driving_city_3")
    all_video.remove("calib_fluorescent_dynamic")
    # These videos are used for sampling testing.
    test_videos = [
        "people_dynamic_wave",
        "indoors_foosball_2",
        "simple_wires_2",
        "people_dynamic_dancing",
        "people_dynamic_jumping",
        "simple_fruit_fast",
        "outdoor_jumping_infrared_2",
        "simple_carpet_fast",
        "people_dynamic_armroll",
        "indoors_kitchen_2",
        "people_dynamic_sitting",
    ]
    # These videos are used for training and testing.
    for i, video in enumerate(all_video):
        if video in test_videos:
            continue
        if i % 8 == 0:
            test_videos.append(video)
        else:
            train_video.append(video)

    train_dataset = ColorEventSRDataset(
        ced_root=config.ced_root,
        videos=train_video,
        in_frame=config.in_frame,
        future_frame=config.future_frame,
        past_frame=config.past_frame,
        scale=config.scale,
        moments=config.moments,
        pyramid_level=config.pyramid_level,
        pyramid_moments=config.pyramid_moments,
        pyramid_reduction_factor=config.pyramid_reduction_factor,
    )
    test_dataset = ColorEventSRDataset(
        ced_root=config.ced_root,
        videos=test_videos,
        in_frame=config.in_frame,
        future_frame=config.future_frame,
        past_frame=config.past_frame,
        scale=config.scale,
        moments=config.moments,
        pyramid_level=config.pyramid_level,
        pyramid_moments=config.pyramid_moments,
        pyramid_reduction_factor=config.pyramid_reduction_factor,
    )
    return train_dataset, test_dataset


class ColorEventSRDataset(Dataset):
    @property
    def height(self):
        return 260

    @property
    def width(self):
        return 346

    def __init__(
        self,
        ced_root,
        videos,
        in_frame,
        future_frame,
        past_frame,
        scale,
        moments,
        pyramid_level,
        pyramid_moments,
        pyramid_reduction_factor,
    ):
        super(ColorEventSRDataset, self).__init__()
        self.ced_root = ced_root
        self.videos = videos
        # Image config
        self.in_frame = in_frame
        self.out_frame = in_frame - future_frame - past_frame
        self.future_frame = future_frame
        self.past_frame = past_frame
        self.scale = scale
        self.high_resolution = (260, 346)
        self.low_resolution = (260 // scale, 346 // scale)
        # Event config
        self.moments = moments
        self.pyramid_level = pyramid_level
        self.pyramid_moments = pyramid_moments
        self.pyramid_reduction_factor = pyramid_reduction_factor
        # Generate the inference and training items.
        self.items = self._generate_items()
        self.deta_t = 0.2

        self.positive = 1
        self.negative = 0

        info(f"ColorEventSRDataset:")
        info(f"  - ced_root: {self.ced_root}")
        info(f"  - number of videos: {len(self.videos)}")
        info(f"  - in_frame: {self.in_frame}")
        info(f"  - scale: {self.scale}")
        info(f"     - up: {self.low_resolution}->{self.high_resolution}")
        info(f"  - moments: {self.moments}")
        info(f"  - event: single polarity")
        info(f"     - positive: {self.positive}")
        info(f"     - negative: {self.negative}")
        info(f"  - items: {len(self.items)}")

    def __getitem__(self, index):
        image_paths, events = self.items[index]
        # Image loading
        timestamps = [get_ced_time_stamp(image) for image in image_paths]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        during_time = end_time - start_time
        video_name = Path(image_paths[0]).parent.name
        frame_names = [Path(image).stem for image in image_paths]
        timestamps_normal = [(t - start_time) / (during_time) for t in timestamps]
        output_norm_timestamp = timestamps_normal[self.past_frame : -self.future_frame]

        #
        images = [Image.open(image) for image in image_paths]
        # Attention, the input of resize function is (width, height)!
        (low_height, low_width) = self.low_resolution
        lr = [img.resize((low_width, low_height)) for img in images]
        hr = images[self.past_frame : -self.future_frame]
        lr = [to_tensor(img) for img in lr]
        hr = [to_tensor(img) for img in hr]
        lr = torch.stack(lr, dim=0)
        hr = torch.stack(hr, dim=0)

        # Event loading
        event_stream = [np.load(event) for event in events]
        event_stream = np.concatenate(event_stream, axis=0)
        event_stream = self._select_events(event_stream, timestamps)
        event_stream[event_stream[:, 3] == 0] = -1

        # events_voxel_grid: M H W
        events_voxel_grid = self._event_stream_to_voxel_grid(event_stream, self.moments, self.high_resolution)
        events_voxel_grid = torch.from_numpy(events_voxel_grid)
        events_voxel_grid = F.interpolate(
            events_voxel_grid.unsqueeze(1),
            size=self.low_resolution,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        events_temporal_pyramid_representations = self._generate_events_temporal_pyramid_representations(
            event_stream,
            output_norm_timestamp,
            self.deta_t,
        )
        # N, PL, PM, H, W
        events_temporal_pyramid_representations = torch.from_numpy(events_temporal_pyramid_representations)
        N, PL, PM, H, W = events_temporal_pyramid_representations.shape
        events_temporal_pyramid_representations = rearrange(
            events_temporal_pyramid_representations, "n pl pm h w -> (n pl pm) 1 h w"
        )
        events_temporal_pyramid_representations = F.interpolate(
            events_temporal_pyramid_representations, size=self.low_resolution, mode="bilinear", align_corners=False
        )
        events_temporal_pyramid_representations = rearrange(
            events_temporal_pyramid_representations, "(n pl pm) 1 h w -> n pl pm h w", n=N, pl=PL, pm=PM
        )

        # make a batch
        batch = get_vfi_rs_batch()
        batch[B.RANDOM_SELECTION] = False
        batch[B.VIDEO_NAME] = video_name
        batch[B.FRAME_NAMES] = frame_names
        batch[B.INPUT_FRAME_COUNT] = self.in_frame
        batch[B.OUTPUT_FRAME_COUNT] = self.in_frame - self.past_frame - self.future_frame
        batch[B.SKIP_FRAME_COUNT] = 0
        #
        batch[B.LOW_FRAME_RATE] = 24
        batch[B.LOW_RESOLUTION] = self.low_resolution
        batch[B.HIGH_FRAME_RATE] = 24
        batch[B.HIGH_RESOLUTION] = self.high_resolution
        #
        batch[B.LFR_LR_FRAMES_TIMESTAMPS_RAW] = torch.FloatTensor(timestamps)
        batch[B.LFR_LR_FRAMES_TIMESTAMPS_NORMAL] = torch.FloatTensor(timestamps_normal)
        batch[B.HFR_HR_FRAMES_TIMESTAMPS_RAW] = torch.FloatTensor(timestamps[self.past_frame : -self.future_frame])
        batch[B.HFR_HR_FRAMES_TIMESTAMPS_NORMAL] = torch.FloatTensor(
            timestamps_normal[self.past_frame : -self.future_frame]
        )

        batch[B.LFR_LR_FRAMES] = lr
        batch[B.EVENTS_GLOBAL_MOMENTS] = events_voxel_grid
        batch[B.HFR_HR_FRAMES] = hr
        batch[B.EVENTS_PYRAMID_REPRESENTATION_MOMENTS] = events_temporal_pyramid_representations
        return batch

    def __len__(self):
        return len(self.items)

    def _select_events(self, event_stream, timestamps):
        begin_time = min(timestamps)
        end_time = max(timestamps)
        left_index = np.searchsorted(event_stream[:, 0], begin_time, side="left")
        right_index = np.searchsorted(event_stream[:, 0], end_time, side="right")
        return event_stream[left_index:right_index]

    def _generate_events_temporal_pyramid_representations(
        self,
        events_stream,
        selected_norm_timestamp,
        deta_t,
    ):
        # start_time = events_stream[:, 0].min()
        # end_time = events_stream[:, 0].max()
        # during_time = end_time - start_time
        # # events_stream[:, 0] = (events_stream[:, 0] - start_time) / during_time
        # info(f"start: {events_stream[:, 0].min()}")
        # info(f"end: {events_stream[:, 0].max()}")
        # events_stream = events_stream[events_stream[:, 0].argsort()]

        # info(f"events_stream: {events_stream.shape}, {events_stream[:100,0]}")

        t = events_stream[:, 0]
        t = t.astype(np.float32)
        start_time = t.min()
        end_time = t.max()
        during_time = end_time - start_time
        t = (t - start_time) / during_time

        H, W = self.high_resolution
        N, PL, PM = self.out_frame, self.pyramid_level, self.pyramid_moments
        event_temporal_pryamid_representations = np.zeros((N, PL, PM, H, W), dtype=np.float32)
        for i, normal_timestamp in enumerate(selected_norm_timestamp):
            # left_t = (normal_timestamp - deta_t) * during_time + start_time
            # right_t = (normal_timestamp + deta_t) * during_time + start_time
            left_t = normal_timestamp - deta_t
            right_t = normal_timestamp + deta_t
            left_index = np.searchsorted(t, left_t, side="left")
            right_index = np.searchsorted(t, right_t, side="right")
            if DEBUG:
                info(f" --- {i} ---")
                info(f" - during_time:{during_time}")
                info(f" - normal_timestamp:{normal_timestamp}")
                info(f" - left_t:{left_t}")
                info(f" - right_t:{right_t}")
                info(f" - left_index:{left_index}")
                info(f" - right_index:{right_index}")
            selected_local_events = events_stream[left_index:right_index]
            event_temporal_pryamid_representations[i] = event_stream_to_temporal_pyramid_representation(
                events=selected_local_events,
                pyramid_level=self.pyramid_level,
                pyramid_moments=self.pyramid_moments,
                reduction_factor=self.pyramid_reduction_factor,
                resolution=self.high_resolution,
            )
        return event_temporal_pryamid_representations

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

    @staticmethod
    def _render(x, y, p, shape):
        # info(f"render: x:max:{x.max()}, min:{x.min()}, y:max:{y.max()}, min:{y.min()}, p:max:{p.max()}, min:{p.min()}")
        if not np.all(np.logical_or(np.isclose(p, 1), np.isclose(p, -1))):
            raise ValueError(f"p must be 1 or -1, but got {p}")
        events = np.zeros(shape=shape, dtype=np.float32)
        events[y, x] = p
        return events

    def _generate_items(self) -> List:
        items = []
        for video_name in self.videos:
            video_folder = join(self.ced_root, video_name)
            video_items = self._generate_from_video(video_folder)
            if len(video_items):
                items.extend(video_items)
        return items

    def _generate_from_video(self, video_folder):
        files = sorted(listdir(video_folder))
        files = [f for f in files if (f.endswith(".png") or f.endswith(".npy"))]
        items = []
        length = len(files)
        for i in range(1, length):
            left = i
            right = -100
            if files[i].endswith(".png"):
                count = 1
                for j in range(i + 1, length):
                    if files[j].endswith(".png"):
                        count += 1
                    else:
                        continue
                    if count == self.in_frame:
                        right = j
                        break
            else:
                continue
            # Generate training item
            # e, i, i, i, e,
            # e, e, i, e, i
            # e, i, e, i, e, i, e
            if length - 2 > right > left > 0 and files[left - 1].endswith(".npy") and files[right + 1].endswith("npy"):
                item = [[], []]
                for k in range(left - 1, right + 2):
                    if files[k].endswith("png"):
                        item[0].append(join(video_folder, files[k]))
                    elif files[k].endswith("npy"):
                        item[1].append(join(video_folder, files[k]))
                    else:
                        pass
                items.append(item)
        return items
