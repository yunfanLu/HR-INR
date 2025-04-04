import torch
from absl.logging import info
from absl.testing import absltest

from egrsdb.models.egvsr.lcb import LightWeightCNNBackbone


class LightWeightCNNBackboneTest(absltest.TestCase):
    def test_inference(self):
        info("LightWeightCNNBackboneTest.test_inference")
        n_feats = 32
        x = torch.randn(2, 32, 128, 128)
        model = LightWeightCNNBackbone(in_channels=n_feats, depth=3)
        y = model(x)
        info(f"y.shape: {y.shape}")
        self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    absltest.main()
