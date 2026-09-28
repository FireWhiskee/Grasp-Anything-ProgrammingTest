"""Measure label overlap between positive and same-scene negative prompts."""

import argparse
import json
import statistics
from collections import defaultdict

from pcghnet.geometry import rotated_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--max-grasps", type=int, default=10)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    by_image = defaultdict(dict)
    for record in records:
        by_image[record["image"]][record["prompt"]] = record

    overlaps = []
    identical = 0
    same_scene = 0
    for record in records:
        same_scene += record.get("negative_type") == "same_scene"
        negative = by_image[record["image"]].get(record["negative_prompt"])
        if negative is None:
            continue
        identical += record["grasps"] == negative["grasps"]
        best = 0.0
        for positive_grasp in record["grasps"][: args.max_grasps]:
            for negative_grasp in negative["grasps"][: args.max_grasps]:
                best = max(best, rotated_iou(positive_grasp, negative_grasp))
        overlaps.append(best)

    result = {
        "records": len(records),
        "same_scene_negatives": same_scene,
        "matched_negative_labels": len(overlaps),
        "identical_grasp_sets": identical,
        "overlap_ge_0.25": sum(value >= 0.25 for value in overlaps),
        "overlap_ge_0.5": sum(value >= 0.5 for value in overlaps),
        "median_best_iou": statistics.median(overlaps),
        "mean_best_iou": sum(overlaps) / max(1, len(overlaps)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
