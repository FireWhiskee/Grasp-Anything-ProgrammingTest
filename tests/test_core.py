import math
import unittest

import torch

from pcghnet.decode import decode_grasps
from pcghnet.losses import negative_prompt_consistency_loss, pcgh_loss
from pcghnet.model import PCGHNet
from pcghnet.targets import build_grasp_targets


class TargetAndDecodeTest(unittest.TestCase):
    def test_target_can_be_decoded(self):
        targets = build_grasp_targets([[[32, 24, 20, 8, 0.3]]], (64, 64), (16, 16))
        outputs = {
            "center": torch.logit(targets["center"].clamp(1e-4, 1 - 1e-4)),
            "offset": targets["offset"],
            "size": targets["size"],
            "angle": targets["angle"],
        }
        grasp = decode_grasps(outputs, (64, 64), top_k=1)[0][0]
        self.assertLess(abs(grasp[0] - 32), 0.1)
        self.assertLess(abs(grasp[1] - 24), 0.1)
        self.assertLess(abs(grasp[2] - 20), 1e-4)
        self.assertLess(abs(grasp[4] - 0.3), 1e-4)

    def test_consistency_loss_rewards_correct_prompt(self):
        target = torch.zeros(1, 1, 4, 4)
        target[:, :, 2, 2] = 1.0
        good = torch.full_like(target, -4.0)
        bad = torch.full_like(target, -4.0)
        good[:, :, 2, 2] = 4.0
        bad[:, :, 2, 2] = -1.0
        self.assertAlmostEqual(float(negative_prompt_consistency_loss(good, bad, target)), 0.0)


class ModelTest(unittest.TestCase):
    def test_forward_and_backward(self):
        torch.manual_seed(1)
        model = PCGHNet(text_encoder="hash", text_dim=32, fpn_dim=16)
        images = torch.randn(2, 3, 64, 64)
        outputs = model(images, ["red cup", "blue box"])
        self.assertEqual(tuple(outputs["center"].shape), (2, 1, 16, 16))
        targets = build_grasp_targets(
            [[[24, 32, 15, 8, math.pi / 6]], [[40, 20, 12, 6, -0.2]]],
            (64, 64),
            (16, 16),
        )
        losses = pcgh_loss(outputs, targets)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))


if __name__ == "__main__":
    unittest.main()
