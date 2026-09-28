import unittest

import torch

from pcghnet.model import PCGHNet


class AblationTest(unittest.TestCase):
    def test_image_only_model_is_prompt_invariant(self):
        model = PCGHNet(
            text_encoder="hash",
            fpn_dim=16,
            pretrained_backbone=False,
            language_conditioning=False,
        ).eval()
        images = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            first = model(images, ["grasp the red mug"])
            second = model(images, ["lift the metal spoon"])
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))
        self.assertIsNone(model.text_encoder)
        self.assertIsNone(model.film)


if __name__ == "__main__":
    unittest.main()
