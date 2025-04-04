import time

import pudb
import torch
from absl.logging import info
from absl.testing import absltest

from egrsdb.models.rstt.layers import EncoderLayer
from egrsdb.utils.model_size import model_size


class EncoderLayerTest(absltest.TestCase):
    def setUp(self):
        model = EncoderLayer(dim=96, depth=8, num_heads=8, num_frames=3, window_size=(4, 4))
        info(f"small model size: {model_size(model)}")
        self.model = model.cuda()
        B, D, C, H, W = 2, 3, 96, 128, 128
        self.inputs = torch.rand((B, D, C, H, W)).cuda()

    def test_inference_time(self):
        pudb.set_trace()

        N = 5
        batch = self.model(self.inputs)
        info(f"batch shape: {batch.shape}")
        # preheat
        torch.cuda.synchronize()
        for i in range(N):
            batch = self.model(self.inputs)
            torch.cuda.synchronize()
        # test time
        torch.cuda.synchronize()
        start = time.time()
        for i in range(N):
            batch = self.model(self.inputs)
            torch.cuda.synchronize()
        end = time.time()
        info(f"Inference with loding model time: {(end - start)*1000 / N}ms")


if __name__ == "__main__":
    absltest.main()
