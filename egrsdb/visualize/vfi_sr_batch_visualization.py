import logging
import os
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pudb
import torch
from absl.logging import debug, flags, info

from egrsdb.datasets.basic_batch import VFI_SR_BATCH as BC
from egrsdb.utils.flow_viz import flow_to_image

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

FLAGS = flags.FLAGS


def _event_to_image(event, path):
    h, w = event.shape
    image = np.zeros((h, w, 3)) + 255
    image[event[:] > 0] = [0, 0, 255]
    image[event[:] < 0] = [255, 0, 0]
    cv2.imwrite(path, image)


def _tensor_to_image(tensor, path):
    image = tensor.permute(1, 2, 0).numpy().astype(np.float32) * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def _hot_map(tensor, path, cmap="hot"):
    plt.figure(figsize=(9, 7.2))
    plt.imshow(tensor, cmap=cmap, interpolation="nearest")
    # plt.clim(0, 0.5)
    plt.axis("off")
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=30)
    plt.savefig(path)
    plt.close()


class VFISRBatchVisualization:
    def __init__(self, config):
        self.saving_folder = join(FLAGS.log_dir, config.folder)
        os.makedirs(self.saving_folder, exist_ok=True)
        self.count = 0
        self.intermediate_visualization = config.intermediate_visualization
        self.show_etpr = config.show_etpr
        info("Init Visualization:")
        info(f"  saving_folder: {self.saving_folder}")
        info(f"  intermediate_visualization: {self.intermediate_visualization}")

    def visualize(self, batch):
        video_names = batch[BC.VIDEO_NAME]
        batch_size = len(batch[BC.FRAME_NAMES])
        for b in range(1):
            video_name = video_names[b]
            frame_names = batch[BC.FRAME_NAMES][b]
            events = batch[BC.EVENTS_GLOBAL_MOMENTS][b].cpu()
            inputs = batch[BC.LFR_LR_FRAMES][b].cpu()
            gts = batch[BC.HFR_HR_FRAMES][b].cpu()
            outputs = batch[BC.HFR_HR_FRAMES_PRED][b].cpu()
            etprs = batch[BC.EVENTS_PYRAMID_REPRESENTATION_MOMENTS][b]
            if isinstance(etprs, torch.Tensor):
                etprs = etprs.cpu()

            folder = join(self.saving_folder, video_name)
            os.makedirs(folder, exist_ok=True)

            N_out = len(gts)
            for j in range(N_out):
                gt = gts[j]
                out = outputs[j]
                dif = torch.abs(gt - out).mean(dim=0)
                _tensor_to_image(gt, join(folder, f"{frame_names}-{j}-gt.png"))
                _tensor_to_image(out, join(folder, f"{frame_names}-{j}-out.png"))
                _hot_map(dif, join(folder, f"{frame_names}-{j}-dif.png"))
                # etpr
                if self.show_etpr:
                    if isinstance(etprs, torch.Tensor):
                        N_out, PL, PM, H, W = etprs.shape
                        for k in range(PL):
                            etpr_l = etprs[j, k]
                            etpr_l = torch.mean(etpr_l, dim=0)
                            _event_to_image(etpr_l, join(folder, f"{frame_names}-{j}-etpr-l{k}.png"))
                # intermediate visualization

            N_in = len(inputs)
            for j in range(N_in):
                _tensor_to_image(inputs[j], join(folder, f"{frame_names}-input-{j}.png"))

            events = torch.mean(events, dim=0)
            _event_to_image(events, join(folder, f"{frame_names}-event-mean.png"))

            if self.intermediate_visualization:
                regional_features = batch[BC.REGIONAL_EVENT_FEATURES].cpu()
                B, N, H, W = regional_features.shape
                for j in range(N):
                    regional_feature = regional_features[0, j, :, :]
                    _hot_map(regional_feature, join(folder, f"{frame_names}-{j}-regional-feature.png"), cmap="viridis")
                holistic_features = batch[BC.HOLISTIC_EVENT_FRAME_FEATURES].cpu()[0, :, :]
                _hot_map(holistic_features, join(folder, f"{frame_names}-holistic-feature.png"), cmap="viridis")
