import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def _parse_grasp(grasp):
    if isinstance(grasp, dict):
        return [grasp[key] for key in ("x", "y", "w", "h", "theta")]
    if len(grasp) != 5:
        raise ValueError("Each grasp must contain x, y, w, h, theta")
    return list(grasp)


def resize_and_pad(image, grasps, image_size):
    original_width, original_height = image.size
    scale = min(float(image_size) / original_width, float(image_size) / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    image = image.resize((resized_width, resized_height), Image.BILINEAR)
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    canvas.paste(image, (pad_x, pad_y))
    transformed = []
    for x, y, width, height, theta in grasps:
        transformed.append(
            [x * scale + pad_x, y * scale + pad_y, width * scale, height * scale, theta]
        )
    metadata = {
        "original_size": [original_height, original_width],
        "scale": scale,
        "padding": [pad_x, pad_y],
    }
    return canvas, transformed, metadata


class GraspManifestDataset(Dataset):
    """JSONL dataset adapter independent of the original annotation storage."""

    def __init__(self, manifest, image_size=224, normalize=True):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.image_size = image_size
        self.normalize = normalize
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.samples = [json.loads(line) for line in handle if line.strip()]
        if not self.samples:
            raise ValueError("Manifest is empty: {}".format(self.manifest))

    def __len__(self):
        return len(self.samples)

    def _negative_prompt(self, index, sample):
        if sample.get("negative_prompt"):
            return sample["negative_prompt"]
        if len(self.samples) == 1:
            return "grasp a different object"
        other_index = random.randrange(len(self.samples) - 1)
        if other_index >= index:
            other_index += 1
        return self.samples[other_index]["prompt"]

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = Path(sample["image"])
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(str(image_path)).convert("RGB")
        grasps = [_parse_grasp(grasp) for grasp in sample["grasps"]]
        image, grasps, metadata = resize_and_pad(image, grasps, self.image_size)
        tensor = TF.to_tensor(image)
        if self.normalize:
            tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return {
            "image": tensor,
            "prompt": sample["prompt"],
            "negative_prompt": self._negative_prompt(index, sample),
            "grasps": grasps,
            "metadata": metadata,
            "image_path": str(image_path),
        }


def collate_grasp_batch(samples):
    return {
        "images": torch.stack([sample["image"] for sample in samples]),
        "prompts": [sample["prompt"] for sample in samples],
        "negative_prompts": [sample["negative_prompt"] for sample in samples],
        "grasps": [sample["grasps"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "image_paths": [sample["image_path"] for sample in samples],
    }
