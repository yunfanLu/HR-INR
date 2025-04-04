import numpy as np
import torch
from absl import logging
from absl.logging import info
from absl.testing import absltest
from pudb import set_trace
from thop import profile
from torch.utils.data import DataLoader

from config import global_path as gp
from egrsdb.core.launch import move_tensors_to_cuda
from egrsdb.datasets import get_gev_rolling_shutter_blur_dataset
from egrsdb.models.els_net.els_net import ESL


def model_size(network):
    t = sum([p.data.nelement() for p in network.parameters() if p.requires_grad is True])
    t = t / 1024
    t = t / 1024
    return f"{t:.6f} M"


class ESLTest(absltest.TestCase):
    def test_color_inference(self):
        event_moment = 40
        gs_sharp_count = 1
        self.test, self.train = get_gev_rolling_shutter_blur_dataset(
            root=gp.fps5000_video_folder,
            blur_accumulate=260,
            events_moment=event_moment,
            gs_sharp_frame_count=gs_sharp_count,
            center_cropped_height=256,
            random_cropped_width=256,
            is_color=False,
            gs_sharp_start_index=0,
            gs_sharp_end_index=520,
            calculate_in_linear_domain=False,
            event_for_gs_frame_buffer=5,
            correct_offset=True,
        )
        test_loader = DataLoader(self.test, batch_size=1, shuffle=True, num_workers=1)
        self.net_2 = ESL(1, is_color=False).cuda()
        info(f"model size color_1x : {model_size(self.net_2)}")

        output = None
        for i, sample in enumerate(test_loader):
            batch = move_tensors_to_cuda(sample)
            output = self.net_2(batch)

            video_name = output["video_name"]

            rolling_blur_frame_name = output["rolling_blur_frame_name"]
            events = output["events"]
            events_for_gs_sharp_frames = output["events_for_gs_sharp_frames"]
            rolling_blur_frame_color = output["rolling_blur_frame_color"]
            rolling_blur_frame_gray = output["rolling_blur_frame_gray"]
            rolling_blur_start_time = output["rolling_blur_start_time"]
            rolling_blur_end_time = output["rolling_blur_end_time"]
            rolling_blur_exposure_time = output["rolling_blur_exposure_time"]
            rolling_sharp_pred_frames = output["rolling_sharp_pred_frames"]
            rolling_blur_pred_frame = output["rolling_blur_pred_frame"]
            global_sharp_frame_timestamps = output["global_sharp_frame_timestamps"]
            global_sharp_frames = output["global_sharp_frames"]
            global_sharp_pred_frames = output["global_sharp_pred_frames"]
            global_sharp_pred_frames_differential = output["global_sharp_pred_frames_differential"]

            info(f"video_name: {video_name}")
            info(f"rolling_blur_frame_name: {rolling_blur_frame_name}")
            info(f"events: {events.shape}")
            info(f"events_for_gs_sharp_frames: {events_for_gs_sharp_frames.shape}")
            info(f"rolling_blur_frame_color: {rolling_blur_frame_color}")
            info(f"rolling_blur_frame_gray: {rolling_blur_frame_gray.shape}")
            info(f"rolling_blur_start_time: {rolling_blur_start_time}")
            info(f"rolling_blur_end_time: {rolling_blur_end_time}")
            info(f"rolling_blur_exposure_time: {rolling_blur_exposure_time}")
            info(f"rolling_sharp_pred_frames: {rolling_sharp_pred_frames}")
            info(f"rolling_blur_pred_frame: {rolling_blur_pred_frame}")
            info(f"global_sharp_frame_timestamps: {global_sharp_frame_timestamps}")
            info(f"global_sharp_frames: {global_sharp_frames.shape}")
            info(f"global_sharp_pred_frames: {global_sharp_pred_frames.shape}")
            info(f"global_sharp_pred_frames_differential: {global_sharp_pred_frames_differential}")

            break


if __name__ == "__main__":
    absltest.main()
