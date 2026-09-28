import unittest

import torch

from pcghnet.losses import pcgh_loss
from pcghnet.model import PCGHNet
from pcghnet.targets import build_grasp_targets


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP validation")
class AMPTrainingTest(unittest.TestCase):
    def test_first_scaled_step_updates_parameters(self):
        torch.manual_seed(3)
        model = PCGHNet(text_encoder="hash", text_dim=32, fpn_dim=16).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(init_scale=1024.0)
        images = torch.randn(2, 3, 64, 64, device="cuda")
        grasps = [[[24, 32, 15, 8, 0.2]], [[40, 20, 12, 6, -0.2]]]
        tracked = model.heads["center"].weight
        before = tracked.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            features = model.encode_image(images)
            outputs = model.predict_from_features(features, ["red cup", "blue box"])
            negative = model.predict_from_features(features, ["blue box", "red cup"])
            targets = build_grasp_targets(
                grasps, images.shape[-2:], outputs["center"].shape[-2:], device="cuda"
            )
            losses = pcgh_loss(outputs, targets, negative)
        scale_before = scaler.get_scale()
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        self.assertGreaterEqual(scaler.get_scale(), scale_before)
        self.assertFalse(torch.equal(before, tracked.detach()))


if __name__ == "__main__":
    unittest.main()
