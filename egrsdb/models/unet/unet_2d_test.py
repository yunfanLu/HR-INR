#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/5 11:07
from logging import info

import torch
from absl.testing import absltest
from thop import profile

from egrsdb.models.unet.unet_2d import UNet


class UNestTest(absltest.TestCase):
    def test_unet_2d(self):
        image_hight = 256
        image_width = 346 // 32 * 32
        x = torch.Tensor(1, 6, image_hight, image_width).cuda()
        info("x size: {}".format(x.size()))
        model = UNet(in_channels=6, out_channels=12).cuda()
        out = model(x)
        info("out size: {}".format(out.size()))
        flops, params = profile(model, inputs=(x,))  # type: ignore
        info("flops: {}G".format(flops / 1024**3))
        info("params: {}M".format(params / 1024**2))


if __name__ == "__main__":
    absltest.main()
