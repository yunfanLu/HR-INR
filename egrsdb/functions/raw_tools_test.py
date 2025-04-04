from absl.testing import absltest
from absl.logging import info
import torch

from egrsdb.functions.raw_tools import quad_bayes_to_bayes


class RAWToolTest(absltest.TestCase):
    def test_quad_bayes_to_bayes(self):
        raw = [
            [
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [2, 2, 0, 0],
                    [2, 2, 0, 0],
                ]
            ]
        ]
        raw = torch.tensor(raw)
        info(f"raw: \n{raw}")
        new_raw = quad_bayes_to_bayes(raw)
        info(f"new_raw: \n{new_raw}")


if __name__ == "__main__":
    absltest.main()
