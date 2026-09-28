"""Render PCGH-Net predictions and ground-truth grasps for qualitative analysis."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from pcghnet.checkpoint import load_model
from pcghnet.dataset import GraspManifestDataset
from pcghnet.decode import decode_grasps
from pcghnet.geometry import rectangle_corners


def draw_grasp(draw, grasp, color, width):
    corners = rectangle_corners(grasp)
    draw.line(corners + [corners[0]], fill=color, width=width, joint="curve")
    draw.line([corners[0], corners[1]], fill=(255, 255, 255), width=width)


def denormalize_image(tensor):
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    array = (
        (tensor * std + mean)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-ground-truth", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GraspManifestDataset(args.manifest, image_size=args.image_size)
    model, _ = load_model(args.checkpoint, args.device)
    model.eval()
    index_rows = []

    for index in range(min(args.max_samples, len(dataset))):
        sample = dataset[index]
        image_tensor = sample["image"].unsqueeze(0).to(args.device)
        outputs = model(image_tensor, [sample["prompt"]])
        predictions = decode_grasps(outputs, image_tensor.shape[-2:], top_k=1)[0]
        image = denormalize_image(sample["image"])
        draw = ImageDraw.Draw(image)
        for grasp in sample["grasps"][: args.max_ground_truth]:
            draw_grasp(draw, grasp, (40, 220, 90), 2)
        if predictions:
            draw_grasp(draw, predictions[0], (235, 55, 55), 3)
        filename = "sample_{:04d}.png".format(index)
        image.save(output_dir / filename)
        index_rows.append(
            {
                "file": filename,
                "prompt": sample["prompt"],
                "prediction": predictions[0] if predictions else None,
                "legend": {"prediction": "red", "ground_truth": "green"},
            }
        )

    with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index_rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("wrote {} visualizations to {}".format(len(index_rows), output_dir))


if __name__ == "__main__":
    main()
