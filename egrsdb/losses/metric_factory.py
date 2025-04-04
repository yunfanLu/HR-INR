import torch
from absl.logging import info
from torch import nn

from egrsdb.losses.demosic_loss import (
    DemosaicImageReconstructionLPIPS,
    DemosaicImageReconstructionPSNR,
    DemosaicImageReconstructionSSIM,
)
from egrsdb.losses.global_shutter_reconstruct import GlobalShutterReconstructedMetric
from egrsdb.losses.lpips import LPIPS
from egrsdb.losses.psnr import _PSNR
from egrsdb.losses.ssim import SSIM
from egrsdb.losses.vfi_sr_reconstruct import VFISRReconstructedMetric
from egrsdb.losses.image_loss import RGBEISPLoss


def get_single_metric(config):
    # vfi + sr
    # PSNR
    if config.NAME == "vfi_sr-PSNR-wo_input_frames":
        return VFISRReconstructedMetric(metric=_PSNR(), remove_input_frames=True)
    elif config.NAME == "vfi_sr-PSNR-w_input_frames-gray":
        return VFISRReconstructedMetric(metric=_PSNR(), remove_input_frames=False, to_gray=True)
    elif config.NAME == "vfi_sr-PSNR-w_input_frames":
        return VFISRReconstructedMetric(metric=_PSNR(), remove_input_frames=False)
    #
    elif config.NAME == "vfi_sr-LPIPS-w_input_frames":
        return VFISRReconstructedMetric(metric=LPIPS(), remove_input_frames=False)
        # SSIM
    elif config.NAME == "vfi_sr-SSIM-wo_input_frames":
        return VFISRReconstructedMetric(metric=SSIM(), remove_input_frames=True)
    elif config.NAME == "vfi_sr-SSIM-w_input_frames-gray":
        return VFISRReconstructedMetric(metric=SSIM(), remove_input_frames=False, to_gray=True)
    elif config.NAME == "vfi_sr-SSIM-w_input_frames":
        return VFISRReconstructedMetric(metric=SSIM(), remove_input_frames=False)
    # other
    elif config.NAME == "empty":
        return EmptyMetric(config)
    else:
        raise ValueError(f"Unknown metric: {config.NAME}")


class EmptyMetric(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        info(f"EmptyMetric:")
        info(f"  config: {config}")

    def forward(self, batch):
        return torch.tensor(0.0, requires_grad=True)


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super(MixedMetric, self).__init__()
        self.metric = []
        self.eval = []
        for config in configs:
            self.metric.append(config.NAME)
            self.eval.append(get_single_metric(config))
        info(f"Init Mixed Metric: {configs}")

    def forward(self, batch):
        r = []
        for m, e in zip(self.metric, self.eval):
            r.append((m, e(batch)))
        return r
