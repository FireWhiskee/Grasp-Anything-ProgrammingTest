"""Prepare a compact, scene-disjoint Grasp-Anything++ training subset.

The official language and grasp archives contain millions of tiny files. This
tool reads them in-place and extracts only the selected prompts, labels, and
scene images. The 65 GB image ZIP is exposed as two HTTP-range-readable parts,
so it never has to be downloaded in full.
"""

import argparse
import hashlib
import io
import json
import math
import os
import pickletools
import shutil
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import requests
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_IMAGE_PARTS = (
    (
        "https://hf-mirror.com/datasets/airvlab/Grasp-Anything/resolve/main/image_part_aa",
        34359738368,
    ),
    (
        "https://hf-mirror.com/datasets/airvlab/Grasp-Anything/resolve/main/image_part_ab",
        30653099134,
    ),
)

_SAFE_PROMPT_OPCODES = {
    "PROTO",
    "FRAME",
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "UNICODE",
    "MEMOIZE",
    "BINPUT",
    "LONG_BINPUT",
    "STOP",
}


def safe_load_prompt(payload):
    """Read a string-only pickle without invoking pickle's object loader."""
    prompt = None
    for opcode, argument, _ in pickletools.genops(payload):
        if opcode.name not in _SAFE_PROMPT_OPCODES:
            raise ValueError("Unsafe or unsupported prompt pickle opcode: {}".format(opcode.name))
        if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}:
            if prompt is not None:
                raise ValueError("Prompt pickle contains more than one string")
            prompt = argument
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt pickle does not contain one non-empty string")
    return prompt.strip()


def load_grasp_tensor(payload):
    """Load a tensor-only torch archive using the restricted weights loader."""
    try:
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Preparing official .pt labels requires PyTorch 2.0 or newer") from exc
    if not isinstance(value, torch.Tensor):
        raise ValueError("Grasp label is not a tensor")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError("Expected grasp tensor with shape [N, 6]")
    return value


def tensor_to_grasps(tensor, min_score=0.0, max_grasps=50):
    rows = sorted(tensor.tolist(), key=lambda row: float(row[0]), reverse=True)
    grasps = []
    for score, x, y, width, height, theta_degrees in rows:
        if float(score) < min_score or width <= 0 or height <= 0:
            continue
        theta = (-math.radians(float(theta_degrees)) + math.pi / 2.0) % math.pi - math.pi / 2.0
        grasps.append([float(x), float(y), float(width), float(height), theta])
        if len(grasps) >= max_grasps:
            break
    return grasps


class HTTPRangePart(object):
    def __init__(self, url, size, block_size=512 * 1024, cache_blocks=16):
        self.original_url = url
        self.size = int(size)
        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)
        self.session = requests.Session()
        self.final_url = None
        self.cache = OrderedDict()

    def _resolve(self):
        response = self.session.get(
            self.original_url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            stream=True,
            timeout=60,
        )
        try:
            if response.status_code != 206:
                raise IOError("Server did not honor HTTP range request: {}".format(response.status_code))
            content_range = response.headers.get("Content-Range", "")
            remote_size = int(content_range.rsplit("/", 1)[-1])
            if remote_size != self.size:
                raise IOError("Remote part size changed: {} != {}".format(remote_size, self.size))
            self.final_url = response.url
        finally:
            response.close()

    def _request(self, start, length):
        if self.final_url is None:
            self._resolve()
        end = start + length - 1
        for attempt in range(2):
            response = self.session.get(
                self.final_url,
                headers={
                    "Range": "bytes={}-{}".format(start, end),
                    "Accept-Encoding": "identity",
                },
                timeout=120,
            )
            if response.status_code == 206:
                data = response.content
                if len(data) != length:
                    raise IOError("Short HTTP range read: {} != {}".format(len(data), length))
                return data
            if response.status_code in (401, 403) and attempt == 0:
                self.final_url = None
                self._resolve()
                continue
            raise IOError("HTTP range read failed with status {}".format(response.status_code))
        raise IOError("HTTP range read failed")

    def read_at(self, offset, length):
        if length <= 0 or offset >= self.size:
            return b""
        length = min(length, self.size - offset)
        if length >= self.block_size:
            return self._request(offset, length)

        output = bytearray()
        while length:
            block_index = offset // self.block_size
            block_offset = offset % self.block_size
            if block_index not in self.cache:
                start = block_index * self.block_size
                block_length = min(self.block_size, self.size - start)
                self.cache[block_index] = self._request(start, block_length)
                while len(self.cache) > self.cache_blocks:
                    self.cache.popitem(last=False)
            block = self.cache.pop(block_index)
            self.cache[block_index] = block
            chunk = block[block_offset : block_offset + length]
            output.extend(chunk)
            consumed = len(chunk)
            offset += consumed
            length -= consumed
        return bytes(output)

    def close(self):
        self.session.close()


class SplitRemoteFile(io.RawIOBase):
    def __init__(self, parts):
        super().__init__()
        self.parts = list(parts)
        self.offsets = []
        running = 0
        for part in self.parts:
            self.offsets.append(running)
            running += part.size
        self.size = running
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_CUR:
            offset += self.position
        elif whence == os.SEEK_END:
            offset += self.size
        elif whence != os.SEEK_SET:
            raise ValueError("Unsupported whence")
        if offset < 0:
            raise ValueError("Negative seek position")
        self.position = min(offset, self.size)
        return self.position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        output = bytearray()
        while size > 0:
            part_index = len(self.parts) - 1
            for index, start in enumerate(self.offsets):
                if self.position < start + self.parts[index].size:
                    part_index = index
                    break
            local_offset = self.position - self.offsets[part_index]
            chunk_size = min(size, self.parts[part_index].size - local_offset)
            chunk = self.parts[part_index].read_at(local_offset, chunk_size)
            if not chunk:
                break
            output.extend(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return bytes(output)

    def close(self):
        if not self.closed:
            for part in self.parts:
                part.close()
        super().close()


def select_instruction_members(archive, max_scenes, samples_per_scene):
    selected = OrderedDict()
    for info in tqdm(archive.infolist(), desc="index prompts"):
        if info.is_dir() or not info.filename.endswith(".pkl"):
            continue
        stem = Path(info.filename).stem
        scene_id = stem[:64]
        if scene_id not in selected:
            if len(selected) >= max_scenes:
                continue
            selected[scene_id] = []
        if len(selected[scene_id]) < samples_per_scene:
            selected[scene_id].append(info)
    return selected


def extract_scene_images(scene_ids, output_dir, part_specs):
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [scene for scene in scene_ids if not (output_dir / (scene + ".jpg")).exists()]
    if not missing:
        return
    parts = [HTTPRangePart(url, size) for url, size in part_specs]
    remote_file = SplitRemoteFile(parts)
    try:
        with zipfile.ZipFile(remote_file) as archive:
            infos = []
            for scene_id in missing:
                try:
                    infos.append(archive.getinfo("image/{}.jpg".format(scene_id)))
                except KeyError:
                    print("warning: image missing for scene {}".format(scene_id), file=sys.stderr)
            infos.sort(key=lambda info: info.header_offset)
            for info in tqdm(infos, desc="extract remote images"):
                destination = output_dir / Path(info.filename).name
                with archive.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                with Image.open(str(destination)) as image:
                    image.verify()
    finally:
        remote_file.close()


def _scene_is_validation(scene_id, validation_ratio):
    value = int(hashlib.sha256(scene_id.encode("ascii")).hexdigest()[:8], 16)
    return value / float(0xFFFFFFFF) < validation_ratio


def build_records(instruction_archive, label_archive, selected, images_dir, args):
    records_by_scene = OrderedDict()
    for scene_id, members in tqdm(selected.items(), desc="read prompts and labels"):
        image_path = images_dir / (scene_id + ".jpg")
        if not image_path.exists():
            continue
        scene_records = []
        for member in members:
            stem = Path(member.filename).stem
            label_name = "grasp_label_positive/{}.pt".format(stem)
            try:
                label_info = label_archive.getinfo(label_name)
            except KeyError:
                continue
            prompt = safe_load_prompt(instruction_archive.read(member))
            tensor = load_grasp_tensor(label_archive.read(label_info))
            grasps = tensor_to_grasps(tensor, args.min_score, args.max_grasps)
            if not grasps:
                continue
            scene_records.append(
                {
                    "id": stem,
                    "image": "images/{}.jpg".format(scene_id),
                    "prompt": prompt,
                    "grasps": grasps,
                }
            )
        if scene_records:
            records_by_scene[scene_id] = scene_records

    all_records = [record for records in records_by_scene.values() for record in records]
    for scene_id, records in records_by_scene.items():
        for index, record in enumerate(records):
            if len(records) > 1:
                record["negative_prompt"] = records[(index + 1) % len(records)]["prompt"]
                record["negative_type"] = "same_scene"
            else:
                alternatives = [item for item in all_records if item["prompt"] != record["prompt"]]
                record["negative_prompt"] = (
                    alternatives[0]["prompt"] if alternatives else "grasp a different object"
                )
                record["negative_type"] = "cross_scene"
    return records_by_scene


def write_manifests(records_by_scene, output_root, validation_ratio):
    train_path = output_root / "train.jsonl"
    validation_path = output_root / "val.jsonl"
    counts = {"train": 0, "val": 0, "scenes": len(records_by_scene)}
    with train_path.open("w", encoding="utf-8") as train_handle, validation_path.open(
        "w", encoding="utf-8"
    ) as validation_handle:
        for scene_id, records in records_by_scene.items():
            split = "val" if _scene_is_validation(scene_id, validation_ratio) else "train"
            handle = validation_handle if split == "val" else train_handle
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                counts[split] += 1
    with (output_root / "subset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2)
        handle.write("\n")
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions-zip", required=True)
    parser.add_argument("--labels-zip", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-scenes", type=int, default=1000)
    parser.add_argument("--samples-per-scene", type=int, default=3)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-grasps", type=int, default=50)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument(
        "--image-part",
        action="append",
        help="Override image part as URL,SIZE. Pass once per split part.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    images_dir = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.image_part:
        part_specs = []
        for value in args.image_part:
            url, size = value.rsplit(",", 1)
            part_specs.append((url, int(size)))
    else:
        part_specs = DEFAULT_IMAGE_PARTS

    with zipfile.ZipFile(args.instructions_zip) as instruction_archive:
        selected = select_instruction_members(
            instruction_archive, args.max_scenes, args.samples_per_scene
        )
        if not args.skip_images:
            extract_scene_images(selected.keys(), images_dir, part_specs)
        with zipfile.ZipFile(args.labels_zip) as label_archive:
            records = build_records(
                instruction_archive, label_archive, selected, images_dir, args
            )
    counts = write_manifests(records, output_root, args.validation_ratio)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
