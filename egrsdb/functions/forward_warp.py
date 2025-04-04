#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/4 08:50

from absl.logging import warning
from egvfi.functions.backward_warp import backward_warp
from egvfi.models.softsplat import FunctionSoftsplat
from torch.nn.functional import l1_loss


def forward_warp(image, flow):
    warning(f"this function is not tested, please use with caution.")
    backward_warp_image = backward_warp(image, flow)
    metric = l1_loss(image, backward_warp_image, reduction="none").mean(1, True)
    forward_warp_image = FunctionSoftsplat(backward_warp_image, flow, metric, "softmax")
    return forward_warp_image
