import argparse
import json

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from pcghnet.checkpoint import load_model
from pcghnet.dataset import resize_and_pad
from pcghnet.decode import decode_grasps


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCGH-Net on one image and prompt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model, _ = load_model(args.checkpoint, args.device)
    model.eval()
    original = Image.open(args.image).convert("RGB")
    resized, _, metadata = resize_and_pad(original, [], args.image_size)
    tensor = TF.normalize(
        TF.to_tensor(resized), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ).unsqueeze(0).to(args.device)
    with torch.no_grad():
        outputs = model(tensor, [args.prompt])
    predictions = decode_grasps(outputs, tensor.shape[-2:], top_k=args.top_k)[0]
    pad_x, pad_y = metadata["padding"]
    scale = metadata["scale"]
    for grasp in predictions:
        grasp[0] = (grasp[0] - pad_x) / scale
        grasp[1] = (grasp[1] - pad_y) / scale
        grasp[2] /= scale
        grasp[3] /= scale
    print(json.dumps({"prompt": args.prompt, "grasps": predictions}, indent=2))


if __name__ == "__main__":
    main()
