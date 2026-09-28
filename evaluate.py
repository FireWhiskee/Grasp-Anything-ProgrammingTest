import argparse
import json
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcghnet.checkpoint import load_model
from pcghnet.dataset import GraspManifestDataset, collate_grasp_batch
from pcghnet.decode import decode_grasps
from pcghnet.geometry import is_successful_grasp
from pcghnet.targets import build_grasp_targets


def _target_prompt_gap(positive_logits, negative_logits, target_heatmap):
    region = target_heatmap.gt(0.3).to(positive_logits.dtype)
    count = region.flatten(1).sum(1).clamp_min(1.0)
    positive = (positive_logits.sigmoid() * region).flatten(1).sum(1) / count
    negative = (negative_logits.sigmoid() * region).flatten(1).sum(1) / count
    return positive - negative


@torch.no_grad()
def evaluate_model(model, loader, device, show_progress=True):
    model.eval()
    successes = 0
    evaluated = 0
    overlap_sum = 0.0
    angle_error_sum = 0.0
    prompt_gap_sum = 0.0
    iterator = tqdm(loader, desc="evaluate", leave=False) if show_progress else loader

    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        image_features = model.encode_image(images)
        outputs = model.predict_from_features(image_features, batch["prompts"])
        negative_outputs = model.predict_from_features(
            image_features, batch["negative_prompts"]
        )
        predictions = decode_grasps(outputs, images.shape[-2:], top_k=1)
        targets = build_grasp_targets(
            batch["grasps"],
            input_size=images.shape[-2:],
            output_size=outputs["center"].shape[-2:],
            device=device,
        )
        gaps = _target_prompt_gap(
            outputs["center"], negative_outputs["center"], targets["center"]
        )

        for prediction, ground_truth, gap in zip(predictions, batch["grasps"], gaps):
            if not prediction or not ground_truth:
                continue
            success, overlap, angle_error = is_successful_grasp(
                prediction[0], ground_truth
            )
            successes += int(success)
            evaluated += 1
            overlap_sum += overlap
            angle_error_sum += math.degrees(angle_error)
            prompt_gap_sum += float(gap)

    denominator = max(1, evaluated)
    return {
        "samples": evaluated,
        "success_rate": successes / denominator,
        "mean_best_iou": overlap_sum / denominator,
        "mean_angle_error_degrees": angle_error_sum / denominator,
        "mean_target_prompt_gap": prompt_gap_sum / denominator,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PCGH-Net")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = GraspManifestDataset(args.manifest, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_grasp_batch,
        pin_memory=args.device.startswith("cuda"),
    )
    model, _ = load_model(args.checkpoint, args.device)
    metrics = evaluate_model(model, loader, args.device)
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
