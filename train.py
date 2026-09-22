import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcghnet.checkpoint import save_checkpoint
from pcghnet.dataset import GraspManifestDataset, collate_grasp_batch
from pcghnet.losses import pcgh_loss
from pcghnet.model import PCGHNet
from pcghnet.targets import build_grasp_targets


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCGH-Net")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--text-encoder", choices=["hash", "minilm"], default="minilm")
    parser.add_argument("--no-negative-prompts", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(7)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GraspManifestDataset(args.manifest, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_grasp_batch,
        pin_memory=args.device.startswith("cuda"),
    )
    model = PCGHNet(text_encoder=args.text_encoder, pretrained_backbone=True).to(args.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(loader, desc="epoch {}/{}".format(epoch, args.epochs))
        for batch in progress:
            images = batch["images"].to(args.device, non_blocking=True)
            image_features = model.encode_image(images)
            outputs = model.predict_from_features(image_features, batch["prompts"])
            negative_outputs = None
            if not args.no_negative_prompts:
                negative_outputs = model.predict_from_features(
                    image_features, batch["negative_prompts"]
                )
            targets = build_grasp_targets(
                batch["grasps"],
                input_size=images.shape[-2:],
                output_size=outputs["center"].shape[-2:],
                device=images.device,
            )
            losses = pcgh_loss(outputs, targets, negative_outputs)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            progress.set_postfix(loss="{:.4f}".format(float(losses["total"])))
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch)


if __name__ == "__main__":
    main()
