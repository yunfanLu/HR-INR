import torch
from absl.testing import absltest
from ops_dcnv3.modules import DCNv3


class DCNv3Test(absltest.TestCase):
    def setUp(self):
        channels = 64
        groups = 1
        offset_scale = 2.0
        act_layer = "GELU"
        norm_layer = "LN"
        dw_kernel_size = 3
        center_feature_scale = 0.25
        self.dcn = DCNv3(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
            center_feature_scale=center_feature_scale,
        )  # for InternImage-H/G
        self.dcn = self.dcn.cuda()

    def test_dcnv3(self):
        x = torch.randn(1, 128, 128, 64).cuda()
        y = self.dcn(x)
        self.assertEqual(y.shape, (1, 128, 128, 64))


if __name__ == "__main__":
    absltest.main()
