"""Offline end-to-end check that needs no dataset or downloaded model weights."""

import torch

from pcghnet.decode import decode_grasps
from pcghnet.losses import pcgh_loss
from pcghnet.model import PCGHNet
from pcghnet.targets import build_grasp_targets


def main():
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PCGHNet(text_encoder="hash", text_dim=64, fpn_dim=32).to(device)
    images = torch.randn(2, 3, 96, 96, device=device)
    prompts = ["grasp the red mug", "pick up the silver spoon"]
    negative_prompts = list(reversed(prompts))
    grasps = [[[48, 44, 30, 12, 0.2]], [[61, 52, 36, 8, -0.5]]]

    image_features = model.encode_image(images)
    outputs = model.predict_from_features(image_features, prompts)
    negative_outputs = model.predict_from_features(image_features, negative_prompts)
    targets = build_grasp_targets(
        grasps, images.shape[-2:], outputs["center"].shape[-2:], device=device
    )
    losses = pcgh_loss(outputs, targets, negative_outputs)
    losses["total"].backward()
    decoded = decode_grasps(outputs, images.shape[-2:], top_k=1)

    assert torch.isfinite(losses["total"])
    assert len(decoded) == 2 and len(decoded[0][0]) == 6
    print("smoke test passed")
    print("output shapes:", {key: tuple(value.shape) for key, value in outputs.items()})
    print("losses:", {key: round(float(value.detach()), 4) for key, value in losses.items()})


if __name__ == "__main__":
    main()
