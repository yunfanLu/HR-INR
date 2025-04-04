import io
import random
from os import listdir
from os.path import isdir, join

import lmdb
import numpy as np
import torch
from absl.logging import debug, info, warning
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from egrsdb.datasets.basic_batch import VFI_SR_BATCH as B
from egrsdb.datasets.basic_batch import get_vfi_rs_batch
from egrsdb.functions.rolling_coords import get_t_global_shutter_coordinate
from egrsdb.utils.events_time_pyramid import event_stream_to_temporal_pyramid_representation

DEBUG = False


# the only interface for this dataset
def get_adobe240fps_sr_vfi_dataset(config):
    root = config.root
    train_root = join(root, "train")
    train = Adobe240fpsSTSRDataset(
        only_input_frame=config.only_input_frame,
        is_test_and_visualize=False,
        only_select_middle_frame=config.only_select_middle_frame,
        has_events=config.has_events,
        has_pyramid_representation=config.has_pyramid_representation,
        is_lmdb=config.is_lmdb,
        root=train_root,
        step=config.train_step,
        input_frames=config.input_frames,
        skip_frames=config.skip_frames,
        moments=config.moments,
        pyramid_level=config.pyramid_level,
        pyramid_moments=config.pyramid_moments,
        pyramid_reduction_factor=config.pyramid_reduction_factor,
        random_selection_count=config.random_selection_count,
        is_random_cropping=config.is_random_cropping,
        deta_t_normal=config.deta_t_normal,
        is_random_selection=config.is_random_selection,
        input_resolution=config.input_resolution,
        fixed_up_scale=config.fixed_up_scale,
        crop_size_low_bound=config.crop_size_low_bound,
        crop_size_up_bound=config.crop_size_up_bound,
        is_center_interpolation=config.is_center_interpolation,
        training_with_two_frame_for_ablation=config.training_with_two_frame_for_ablation,
    )
    test_root = join(root, "test")
    test = Adobe240fpsSTSRDataset(
        only_input_frame=config.only_input_frame,
        is_test_and_visualize=config.is_test_and_visualize,
        only_select_middle_frame=config.only_select_middle_frame,
        has_events=config.has_events,
        has_pyramid_representation=config.has_pyramid_representation,
        is_lmdb=config.is_lmdb,
        root=test_root,
        step=config.test_step,
        input_frames=config.input_frames,
        skip_frames=config.skip_frames,
        moments=config.moments,
        random_selection_count=config.random_selection_count,
        pyramid_level=config.pyramid_level,
        pyramid_moments=config.pyramid_moments,
        pyramid_reduction_factor=config.pyramid_reduction_factor,
        deta_t_normal=config.deta_t_normal,
        is_random_cropping=config.is_random_cropping,
        is_random_selection=config.is_random_selection,
        input_resolution=config.input_resolution,
        fixed_up_scale=config.fixed_up_scale,
        crop_size_low_bound=config.crop_size_low_bound,
        crop_size_up_bound=config.crop_size_up_bound,
        is_center_interpolation=config.is_center_interpolation,
        training_with_two_frame_for_ablation=config.training_with_two_frame_for_ablation,
    )
    return train, test


def _adobe240fps_dataset_walks(root, step, input_frames, skip_frames):
    """
    Walks the Adobe240fps dataset.
    :param root: The root folder of Adobe240fps dataset.
    :param step: The step for each item.
    :param skip_frames: The number of frames to skip.
    :return: A list of items.
    For example:
        The original files in each video folder is list as [00000.png, 00001.npy, 00001.png, 00002.npy, ...]
        The output items is as follows:
            [
                [
                    # n is the skip_frames
                    [00001.npy, 00001.npy, 00002.npy ... 0000n.npy], # N items in this list
                    [00000.png, 00001.png, 00002.png ... 0000n.png], # N + 1 items in this list
                ]
            ]
    """
    info(f"Scanning Adobe240fps dataset in {root}")
    videos = listdir(root)
    info(f"Found {len(videos)} videos.")
    items = []
    for video in videos:
        if video in ["IMG_0168"]:
            info(f"Skip {video} for this video is not complete.")
            continue
        video_folder = join(root, video)
        files = sorted(listdir(video_folder))
        png_files = [f for f in files if f.endswith(".png")]
        npy_files = [f for f in files if f.endswith(".npz")]
        png_count = len(png_files)

        frame_count = (input_frames - 1) * (skip_frames + 1) + 1
        for i in range(0, png_count, step):
            if i + frame_count >= png_count:
                break
            item_npy = []
            item_png = []
            for j in range(i, i + frame_count - 1):
                item_npy.append(join(video_folder, npy_files[j]))
                item_png.append(join(video_folder, png_files[j]))
            item_png.append(join(video_folder, png_files[i + frame_count]))
            items.append([item_npy, item_png])
    return items


class Adobe240fpsSTSRDataset(Dataset):
    def __init__(
        self,
        only_input_frame,
        is_test_and_visualize,
        only_select_middle_frame,
        has_events,
        has_pyramid_representation,
        is_lmdb,
        root,
        step,
        input_frames,
        skip_frames,
        moments,
        random_selection_count,
        pyramid_level,
        pyramid_moments,
        pyramid_reduction_factor,
        deta_t_normal,
        is_random_cropping,
        is_random_selection,
        input_resolution,
        fixed_up_scale,
        crop_size_low_bound,
        crop_size_up_bound,
        is_center_interpolation,
        training_with_two_frame_for_ablation,
    ):
        super(Adobe240fpsSTSRDataset, self).__init__()
        # Check the params is correct.
        if is_test_and_visualize:
            assert only_select_middle_frame, f"visualization the middel frame is only support."
            assert (random_selection_count == skip_frames + 2) or (
                random_selection_count == 3
            ), f"when visualization all frame should be selected."
            assert crop_size_low_bound == crop_size_up_bound, f"when visualization the crop should be fixed."
        assert isdir(root), f"Folder is not correct. {root}"
        # todo, check the moments is currect.
        assert is_random_cropping is True, f"Now only support random cropping."
        if has_pyramid_representation:
            assert has_events is True, f"Has pyramid representation must has events."
        assert (
            fixed_up_scale >= 1 or fixed_up_scale == 0
        ), f"Fixed up scale must greater than 1 or equal to 0 (wo fixed)."
        assert not (
            is_random_selection and is_center_interpolation
        ), f"Cannot random selection and center interpolation."
        if fixed_up_scale >= 1:
            ih, iw = input_resolution
            assert crop_size_low_bound == crop_size_up_bound, f"Random crop size must be equal to crop size up bound."
            assert ih * fixed_up_scale == crop_size_up_bound, f"Fixed up scale must be equal to crop size up bound."
            assert iw * fixed_up_scale == crop_size_up_bound, f"Fixed up scale must be equal to crop size up bound."
        # if only_select_middle_frame:
        #     assert input_frames == 4, f"Only select middle frame must be 4 input frames."
        #     assert skip_frames == 7, f"Only select middle frame must be 7 skip frames."
        #     assert random_selection_count <= 9, f"Only select middle frame must be less than 9 random selection count."
        assert pyramid_reduction_factor > 1, f"Pyramid reduction factor must greater than 1."
        if is_center_interpolation:
            assert random_selection_count == 3, f"Center interpolation must be 3 random selection count."
        if only_input_frame:
            assert is_center_interpolation is False, f"Only input frame must be 2 random selection count."
            assert random_selection_count == 2, f"Only input frame must be 2 random selection count."
        # check the stage when training_with_two_frame_for_ablation
        if training_with_two_frame_for_ablation:
            assert only_select_middle_frame, f"Training with two frame for ablation must be only select middle frame."
            assert only_input_frame == False, f"Training with two frame for ablation must be only input frame is False."
            assert (
                is_center_interpolation == False
            ), f"Training with two frame for ablation must be is center interpolation is False."

        info(f"Loading Adobe240 STSR dataset in {root}")
        self.only_input_frame = only_input_frame
        self.is_test_and_visualize = is_test_and_visualize
        self.only_select_middle_frame = only_select_middle_frame
        # this deteminate the events.
        self.has_events = has_events
        self.has_pyramid_representation = has_pyramid_representation
        #
        self.is_lmdb = is_lmdb
        self.root = root
        self.step = step
        self.input_frames = input_frames
        self.skip_frames = skip_frames
        self.gt_frames = (input_frames - 1) * (skip_frames + 1) + 1
        self.is_random_cropping = is_random_cropping
        self.is_random_selection = is_random_selection
        self.random_selection_count = random_selection_count
        self.deta_t = deta_t_normal
        self.moments = moments
        # local event pyramid representation
        self.pyramid_level = pyramid_level
        self.pyramid_moments = pyramid_moments
        self.pyramid_reduction_factor = pyramid_reduction_factor
        self.positive_event = 1
        self.negative_event = -1
        # random cropping
        # the original resolution of Adobe240fps dataset is 1280x720
        self.original_resolution = [720, 1280]
        self.fixed_up_scale = fixed_up_scale
        self.crop_size_low_bound = crop_size_low_bound  # 512
        self.crop_size_up_bound = crop_size_up_bound  # 720
        self.input_resolution = input_resolution
        self.is_center_interpolation = is_center_interpolation

        # for an ablation study
        self.training_with_two_frame_for_ablation = training_with_two_frame_for_ablation

        # data preprocessing
        self.to_tensor = transforms.ToTensor()
        # walk for items
        self._items = _adobe240fps_dataset_walks(self.root, self.step, self.input_frames, self.skip_frames)
        # LMDB
        if self.is_lmdb:
            self.lmdb_root = f"{root}.lmdb"
            assert isdir(self.lmdb_root)
            self.env = lmdb.open(
                self.lmdb_root, subdir=True, max_readers=64, readonly=True, lock=False, readahead=False, meminit=False
            )

        self._info()

    def _info(self):
        info(f"Found {len(self._items)} items in Adobe240fps STSR dataset.")
        info(f"  - Only select middle frame: {self.only_select_middle_frame}")
        info(f"  - Has events: {self.has_events}")
        info(f"  - Has pyramid representation: {self.has_pyramid_representation}")
        info(f"  - Is LMDB: {self.is_lmdb}")
        info(f"  - Root: {self.root}")
        info(f"  - Step: {self.step}")
        info(f"  - Input frames: {self.input_frames}")
        info(f"  - Skip frames: {self.skip_frames}")
        info(f"  - GT frames: {self.gt_frames}")
        info(f"  - Is random cropping: {self.is_random_cropping}")
        info(f"  - Is random selection: {self.is_random_selection}")
        info(f"  - Random selection count: {self.random_selection_count}")
        info(f"  - Moments: {self.moments}")
        info(f"  - Pyramid level: {self.pyramid_level}")
        info(f"  - Pyramid moments: {self.pyramid_moments}")
        info(f"  - Pyramid reduction factor: {self.pyramid_reduction_factor}")
        info(f"  - Positive event: {self.positive_event}")
        info(f"  - Negative event: {self.negative_event}")
        info(f"  - Original resolution: {self.original_resolution}")
        info(f"  - Fixed up scale: {self.fixed_up_scale}")
        info(f"  - Crop size low bound: {self.crop_size_low_bound}")
        info(f"  - Crop size up bound: {self.crop_size_up_bound}")
        info(f"  - Input resolution: {self.input_resolution}")
        info(f"  - Is center interpolation: {self.is_center_interpolation}")

    def _load_image(self, path):
        if self.is_lmdb:
            # remove root in path
            key = path[len(self.root) + 1 :]
            with self.env.begin(write=False) as txn:
                data = txn.get(key.encode())
                if key.endswith(".png"):
                    return Image.open(io.BytesIO(data))
        return Image.open(path)

    def _load_npz(self, path):
        if self.is_lmdb:
            key = path[len(self.root) + 1 :]
            with self.env.begin(write=False) as txn:
                data = txn.get(key.encode())
                with io.BytesIO(data) as f:
                    npz_data = np.load(f)
                    # 返回一个字典，其中键是数组的名称，值是数组数据
                    return {name: npz_data[name] for name in npz_data.files}
                    # return npz_data
        return np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        npy_list, png_list = self._items[index]
        # load png files
        (
            input_tensors,
            gt_tensors,
            selected_index,
            selected_norm_timestamp,
            need_rotate,
        ) = self._get_input_and_gt_image_tensor(png_list)

        N, C, H, W = input_tensors.shape
        if H != 720 or W != 1280:
            warning(f"Input tensor shape is not correct. {input_tensors.shape}")
            warning(f"npy_list: {npy_list}")
            warning(f"png_list: {png_list}")
            raise ValueError("Input tensor shape is not correct.")

        if DEBUG:
            info(f"Sample {index} has {len(npy_list)} npy files and {len(png_list)} png files.")
            # for i, png in enumerate(png_list):
            #     info(f"    {i}: {png}")
            info("----------")
            info(f" -- input_tensors: {input_tensors.shape}")
            info(f" -- gt_tensors: {gt_tensors.shape}")
            info(f" -- selected_index: {selected_index}")
            info(f" -- selected_norm_timestamp: {selected_norm_timestamp}")
            info(f" -- need_rotate: {need_rotate}")

        # if the events need be rotated, the events is rotated.
        if self.has_events:
            events_stream = np.concatenate([self._load_events_as_txyp(npy, need_rotate) for npy in npy_list], axis=0)
        else:
            events_stream = None
        # events_stream: [N, 4] and retate is currect
        event_voxel_grids = self._event_stream_to_voxel_grid(events_stream, self.moments, self.original_resolution)
        event_voxel_grids = torch.from_numpy(event_voxel_grids)

        if self.has_events and self.has_pyramid_representation:
            # load events for mutil scale and mutil resolution representation
            event_tpr_tensor = self._generate_events_temporal_pyramid_representations(
                events_stream, selected_norm_timestamp
            )
            event_tpr_tensor = torch.from_numpy(event_tpr_tensor)
        else:
            event_tpr_tensor = "NO-PYRAMID-REPRESENTATION"

        # resolution random cropping
        input_tensors, gt_tensors, event_voxel_grids, event_tpr_tensor, crop_resolution = self._random_crop(
            input_tensors, gt_tensors, event_voxel_grids, event_tpr_tensor
        )

        if DEBUG:
            info("---------- After Crop")
            info(f" -- input_tensors: {input_tensors.shape}")
            info(f" -- gt_tensors: {gt_tensors.shape}")
            info(f" -- event_voxel_grids: {event_voxel_grids.shape}")
            info(f" -- event_tpr_tensor: {event_tpr_tensor.shape}")
            info(f" -- crop_resolution: {crop_resolution}")

        # video superresolution random scale resize
        input_tensors, gt_tensors, event_voxel_grids, event_tpr_tensor = self._random_scale_resize(
            input_tensors, gt_tensors, event_voxel_grids, event_tpr_tensor
        )
        # generate LFR_LR_FRAMES_TIMESTAMPS_COORDS
        # selected_query_coords = self._generate_query_coords(selected_norm_timestamp, up_scale)
        # ------------------- make data for training ------------------- #
        # generate a batch
        batch = get_vfi_rs_batch()
        batch[B.RANDOM_SELECTION] = self.is_random_selection
        batch[B.VIDEO_NAME] = png_list[0].split("/")[-2]
        batch[B.FRAME_NAMES] = png_list[0].split("/")[-1].split(".")[0]
        batch[B.OUTPUT_FRAME_COUNT] = len(png_list) - 2
        batch[B.EVENTS_GLOBAL_MOMENTS] = event_voxel_grids
        batch[B.LFR_LR_FRAMES] = input_tensors.to(dtype=torch.float)
        batch[B.LOW_FRAME_RATE] = 240 / (self.skip_frames + 1)
        batch[B.LOW_RESOLUTION] = self.input_resolution
        batch[B.LFR_LR_FRAMES_TIMESTAMPS_RAW] = [0, self.skip_frames + 1]
        batch[B.LFR_LR_FRAMES_TIMESTAMPS_NORMAL] = [0, 1]
        batch[B.HFR_HR_FRAMES] = gt_tensors.to(dtype=torch.float)
        batch[B.HIGH_FRAME_RATE] = 240
        batch[B.HIGH_RESOLUTION] = crop_resolution, crop_resolution
        batch[B.HFR_HR_FRAMES_TIMESTAMPS_RAW] = torch.FloatTensor(selected_index)
        batch[B.HFR_HR_FRAMES_TIMESTAMPS_NORMAL] = torch.FloatTensor(selected_norm_timestamp)
        batch[B.EVENTS_PYRAMID_REPRESENTATION_MOMENTS] = event_tpr_tensor
        return batch

    def _get_input_and_gt_image_tensor(self, png_list):
        # load png files
        N = len(png_list)
        # random selection
        selected_index = self._interpolation_index_selection(N)
        selected_norm_timestamp = [index / self.gt_frames for index in selected_index]
        # GT Tensor
        gt_tensors = [self.to_tensor(self._load_image(png_list[i])) for i in selected_index]
        # Input Tensor
        # input and ground truth, input_tensor: [2, 3, H, W], gt_tensor: [skip_frames - 2, 3, H, W]
        input_tensors = []
        for i in range(self.input_frames):
            png = png_list[i * (self.skip_frames + 1)]
            input_tensor = self.to_tensor(self._load_image(png))
            input_tensors.append(input_tensor)

        if self.training_with_two_frame_for_ablation == 2:
            # This flag is test for the ablation study with two frame as inputs.
            # 0 1 2 3 -> 1 1 2 2
            input_tensors[0] = input_tensors[1]
            input_tensors[3] = input_tensors[2]
        elif self.training_with_two_frame_for_ablation == 3:
            input_tensors[0] = input_tensors[1]

        # some image is 1280x720 not 720x1280. So we need to rotate it.
        H, W = gt_tensors[0].shape[-2:]
        need_rotate = H == 1280 and W == 720
        if need_rotate:
            gt_tensors = [gt_tensor.permute(0, 2, 1) for gt_tensor in gt_tensors]
            input_tensors = [input_tensor.permute(0, 2, 1) for input_tensor in input_tensors]

        input_tensors = torch.stack(input_tensors, dim=0)
        gt_tensors = torch.stack(gt_tensors, dim=0)

        return input_tensors, gt_tensors, selected_index, selected_norm_timestamp, need_rotate

    def _interpolation_index_selection(self, N):
        if self.only_select_middle_frame:
            # The input frames are 4.
            # The skip frames are 7.
            # [(0),1,2,3,4,5,6,7,(8),9,10,11,12,13,14,15,(16),17,18,19,20,21,22,23,(24)]
            if self.only_input_frame:
                the_middle_index = [self.skip_frames + 1, 2 * (self.skip_frames + 1)]
                return the_middle_index

            if self.is_center_interpolation:
                the_middle_index = [8, 12, 16]
                return the_middle_index
            else:
                # the_middle_index = [8, 9, 10, 11, 12, 13, 14, 15, 16]
                the_middle_index = list(range(self.skip_frames + 1, 2 * (self.skip_frames + 1) + 1))
                selected_indexes = sorted(random.sample(the_middle_index, self.random_selection_count))
                return selected_indexes

        if self.is_random_selection:
            if self.is_center_interpolation:
                raise ValueError("Cannot random selection and center interpolation.")
            else:
                selected_indexes = sorted(random.sample(range(N), self.random_selection_count))
        else:
            if self.is_center_interpolation:
                # VideoINR select 3 frames randomly
                selected_indexes = [0, 4, 8]
            else:
                selected_indexes = list(range(N))
        return selected_indexes

    def _random_scale_resize(self, input_tensor, gt_tensor, event_voxel_grids, event_tpr_tensor):
        event_voxel_grids = event_voxel_grids.unsqueeze(0)
        event_voxel_grids = torch.nn.functional.interpolate(
            event_voxel_grids, size=self.input_resolution, mode="bicubic"
        )
        event_voxel_grids = event_voxel_grids.squeeze(0)
        if self.has_events and self.has_pyramid_representation:
            # resize the event_tpr_tensor to the self.input_resolution
            N_out, pl, pm, h, w = event_tpr_tensor.shape
            ih, iw = self.input_resolution
            event_tpr_tensor = event_tpr_tensor.reshape(N_out * pl * pm, 1, h, w)
            event_tpr_tensor = torch.nn.functional.interpolate(
                event_tpr_tensor, size=self.input_resolution, mode="bicubic"
            )
            event_tpr_tensor = event_tpr_tensor.reshape(N_out, pl, pm, ih, iw)

        # Frame Parts
        # resize the input to the self.input_resolution
        input_tensor = torch.nn.functional.interpolate(input_tensor, size=self.input_resolution, mode="bicubic")
        return input_tensor, gt_tensor, event_voxel_grids, event_tpr_tensor

    def _random_crop(self, input_tensor, gt_tensor, event_voxel_grids, event_tpr_tensor):
        OW, OH = self.original_resolution
        if self.is_test_and_visualize:
            cr = self.crop_size_low_bound

            input_tensor = input_tensor[:, :, :cr, :cr]
            gt_tensor = gt_tensor[:, :, :cr, :cr]
            event_voxel_grids = event_voxel_grids[:, :cr, :cr]
            # event_tpr_tensor: N, PL, PM, H, W
            if self.has_events and self.has_pyramid_representation:
                event_tpr_tensor = event_tpr_tensor[:, :, :, :cr, :cr]
            # if cr < OW:
            #     info(f"Resize in random crop: {cr}")
            #     event_voxel_grids = event_voxel_grids.unsqueeze(0)
            #     event_voxel_grids = torch.nn.functional.interpolate(event_voxel_grids, size=(cr, cr), mode="bicubic")
            #     event_voxel_grids = event_voxel_grids.squeeze(0)
            #     input_tensor = torch.nn.functional.interpolate(input_tensor, size=(cr,cr), mode="bicubic")
            #     gt_tensor = torch.nn.functional.interpolate(gt_tensor, size=(cr,cr), mode="bicubic")
            #     N_out, pl, pm, h, w = event_tpr_tensor.shape
            #     event_tpr_tensor = event_tpr_tensor.reshape(N_out * pl * pm, 1, h, w)
            #     event_tpr_tensor = torch.nn.functional.interpolate(event_tpr_tensor, size=(cr,cr), mode="bicubic")
            #     event_tpr_tensor = event_tpr_tensor.reshape(N_out, pl, pm, cr, cr)
            crop_resolution = cr
        else:
            crop_resolution = random.randint(self.crop_size_low_bound, self.crop_size_up_bound)
            # info(f"========Random crop: {crop_resolution}")
            low_x, low_y = random.randint(0, OW - crop_resolution), random.randint(0, OH - crop_resolution)
            # info(f"Random crop: {low_x}, {low_y}")
            up_x, up_y = low_x + crop_resolution, low_y + crop_resolution
            # info(f"Random crop: {up_x}, {up_y}")
            input_tensor = input_tensor[:, :, low_x:up_x, low_y:up_y]
            gt_tensor = gt_tensor[:, :, low_x:up_x, low_y:up_y]
            # Events are as follows:
            event_voxel_grids = event_voxel_grids[:, low_x:up_x, low_y:up_y]
            # event_tpr_tensor: N, PL, PM, H, W
            if self.has_events and self.has_pyramid_representation:
                event_tpr_tensor = event_tpr_tensor[:, :, :, low_x:up_x, low_y:up_y]
        #
        return input_tensor, gt_tensor, event_voxel_grids, event_tpr_tensor, crop_resolution

    def _load_events_as_txyp(self, event_path, need_rotate):
        events = self._load_npz(event_path)
        events = self._render_to_txyp(need_rotate=need_rotate, **events)
        events = events[events[:, 0].argsort()]
        return events

    @staticmethod
    def _render_to_txyp(x, y, t, p, need_rotate):
        if need_rotate:
            x, y = y, x
        events = np.stack([t, x, y, p], axis=-1)
        return events

    @staticmethod
    def _render(x, y, p, shape, need_rotate):
        if need_rotate:
            x, y = y, x
        # info(f"render: x:max:{x.max()}, min:{x.min()}, y:max:{y.max()}, min:{y.min()}, p:max:{p.max()}, min:{p.min()}")
        events = np.zeros(shape=shape, dtype=np.float32)
        events[y, x] = p
        return events

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

        if events_stream is None:
            return voxel_grid

        # The voxel grid is a 3D image.
        start_time = events_stream[:, 0].min()
        end_time = events_stream[:, 0].max()
        voxel_grid_time_step = (end_time - start_time) / moments
        for i in range(moments):
            left_time = start_time + i * voxel_grid_time_step
            right_time = start_time + (i + 1) * voxel_grid_time_step
            left_time_index = np.searchsorted(events_stream[:, 0], left_time)
            right_time_index = np.searchsorted(events_stream[:, 0], right_time)
            lti, rti = left_time_index, right_time_index
            # The voxel grid in a moment is a 2D image.
            x, y, p = events_stream[lti:rti, 1], events_stream[lti:rti, 2], events_stream[lti:rti, 3]
            x = x.astype(np.int32)
            y = y.astype(np.int32)
            voxel_grid[i] = self._render(x=x, y=y, p=p, shape=resolution, need_rotate=False)
        return voxel_grid

    def _generate_events_temporal_pyramid_representations(self, events_stream, selected_norm_timestamp):
        H, W = self.original_resolution
        N, PL, PM = self.random_selection_count, self.pyramid_level, self.pyramid_moments
        event_temporal_pryamid_representations = np.zeros((N, PL, PM, H, W), dtype=np.float32)
        if events_stream is None:
            return event_temporal_pryamid_representations

        deta_t = self.deta_t

        start_time = events_stream[:, 0].min()
        end_time = events_stream[:, 0].max()
        during_time = end_time - start_time
        for i, normal_timestamp in enumerate(selected_norm_timestamp):
            left_t = (normal_timestamp - deta_t) * during_time + start_time
            right_t = (normal_timestamp + deta_t) * during_time + start_time
            left_index = np.searchsorted(events_stream[:, 0], left_t)
            right_index = np.searchsorted(events_stream[:, 0], right_t)
            if DEBUG:
                info(f"i: {i}, normal_timestamp: {normal_timestamp}, left_t: {left_t}, right_t: {right_t}")
                info(f"  left_index: {left_index}, right_index: {right_index}, len: {right_index - left_index}")
            # events, pyramid_level, pyramid_moments, reduction_factor, resolution
            event_temporal_pryamid_representations[i] = event_stream_to_temporal_pyramid_representation(
                events=events_stream[left_index:right_index],
                pyramid_level=self.pyramid_level,
                pyramid_moments=self.pyramid_moments,
                reduction_factor=self.pyramid_reduction_factor,
                resolution=self.original_resolution,
            )
        return event_temporal_pryamid_representations

    # def _generate_event_pyramid_representation(self, npy_list, selected_norm_timestamp, deta_t, need_rotate):
    #     events_npy = [self._load_npz(npy) for npy in npy_list]
    #     events_list = []
    #     for i, event in enumerate(events_npy):
    #         events_list.append(
    #             self._render_to_txyp(**event, need_rotate=need_rotate, set_fixed_time=False, fixed_time=0)
    #         )
    #     events_txyp = np.concatenate(events_list, axis=0).astype(np.float64)
    #     # to float
    #     min_time = np.min(events_txyp[:, 0])
    #     max_time = np.max(events_txyp[:, 0])
    #     events_txyp[:, 0] = (events_txyp[:, 0] - min_time) / (max_time - min_time)

    #     # if DEBUG:
    #     #     debug(f"Event: {events_txyp.shape}")
    #     #     debug(f"Event time: {min_time} - {max_time}")
    #     #     debug(f"Event time: {min(events_txyp[:, 0])} - {max(events_txyp[:, 0])}")
    #     #     t = events_txyp[:, 0]
    #     #     video_name = npy_list[0].split("/")[-2]
    #     #     first_npy_name = npy_list[0].split("/")[-1].split(".")[0]
    #     #     last_npy_name = npy_list[-1].split("/")[-1].split(".")[0]
    #     #     plot_histogram(t, f"test_histogram-{video_name}-{first_npy_name}-{last_npy_name}.png")

    #     events_pyramid_representation_all = []
    #     for timestamp in selected_norm_timestamp:
    #         left_timestamp = timestamp - deta_t
    #         right_timestamp = timestamp + deta_t
    #         local_events = events_txyp[events_txyp[:, 0] >= left_timestamp]
    #         local_events = local_events[local_events[:, 0] < right_timestamp]
    #         # events, pyramid_level, pyramid_moments, pyramid_reduction_factor, resolution
    #         local_events_pyramid_representation = event_stream_to_temporal_pyramid_representation(
    #             events=local_events,
    #             pyramid_level=self.pyramid_level,
    #             pyramid_moments=self.pyramid_moments,
    #             pyramid_reduction_factor=self.pyramid_reduction_factor,
    #             resolution=self.original_resolution,
    #         )
    #         # local_events_pyramid_representation: [pyramid_level, pyramid_moments, 2, H, W]
    #         events_pyramid_representation_all.append(local_events_pyramid_representation)
    #     events_pyramid_representation_all = np.stack(events_pyramid_representation_all, axis=0)
    #     event_tpr_tensor = torch.from_numpy(events_pyramid_representation_all).to(dtype=torch.float)
    #     return event_tpr_tensor
