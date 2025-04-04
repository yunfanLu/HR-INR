import torch
from absl.logging import info
from torch.nn.modules.loss import _Loss

from egrsdb.losses.image_loss import (
    L1CharbonnierLossColor,
)
from egrsdb.losses.pixel_focus_loss import PixelFocusLoss, get_pixel_focus_loss
from egrsdb.losses.vfi_sr_reconstruct import VFISRReconstructedMetric


def get_single_loss(config):
    if config.NAME == "vfi_sr-loss-wo_input_frames":
        return VFISRReconstructedMetric(metric=L1CharbonnierLossColor(), remove_input_frames=True)
    elif config.NAME == "vfi_sr-loss-w_input_frames":
        return VFISRReconstructedMetric(metric=L1CharbonnierLossColor(), remove_input_frames=False)
    else:
        raise ValueError(f"Unknown loss: {config.NAME}")


class EmptyLoss(_Loss):
    def __init__(self, config=None):
        super().__init__()
        info(f"Empty Loss:")
        info(f"  config:{config}")

    def forward(self, batch):
        return torch.tensor(0.0, requires_grad=True)


class MixedLoss(_Loss):
    def __init__(self, configs):
        super(MixedLoss, self).__init__()
        self.loss = []
        self.weight = []
        self.criterion = []
        for item in configs:
            self.loss.append(item.NAME)
            self.weight.append(item.WEIGHT)
            self.criterion.append(get_single_loss(item))
        info(f"Init Mixed Loss: {configs}")

    def forward(self, batch):
        name_to_loss = []
        total = 0
        for n, w, fun in zip(self.loss, self.weight, self.criterion):
            tmp = fun(batch)
            name_to_loss.append((n, tmp))
            total = total + tmp * w
        return total, name_to_loss
