import unittest

import torch

from evaluate import _target_prompt_gap


class PromptGapTest(unittest.TestCase):
    def test_positive_prompt_scores_target_higher(self):
        target = torch.zeros(2, 1, 4, 4)
        target[:, :, 1, 2] = 1.0
        positive = torch.full_like(target, -2.0)
        negative = torch.full_like(target, -2.0)
        positive[:, :, 1, 2] = 3.0
        negative[:, :, 1, 2] = -1.0
        gaps = _target_prompt_gap(positive, negative, target)
        self.assertTrue(torch.all(gaps > 0.0))


if __name__ == "__main__":
    unittest.main()
