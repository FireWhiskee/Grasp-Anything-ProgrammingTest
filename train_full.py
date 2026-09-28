"""Full PCGH-Net training entry point with AMP, validation, and resume support."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate_model
from pcghnet.checkpoint import load_model, save_checkpoint
from pcghnet.dataset import GraspManifestDataset, collate_grasp_batch
from pcghnet.losses import pcgh_loss
from pcghnet.model import PCGHNet
from pcghnet.targets import build_grasp_targets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest")
    parser.add_argument("--output-dir", default="outputs/pcghnet")
    parser.add_argument("--resume")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, help="Limit steps per epoch for debugging")
    parser.add_argument("--text-encoder", choices=["hash", "minilm"], default="minilm")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--no-negative-prompts", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(manifest, args, shuffle):
    dataset = GraspManifestDataset(manifest, image_size=args.image_size)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        collate_fn=collate_grasp_batch,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch):
    model.train()
    running_loss = 0.0
    steps = 0
    progress = tqdm(loader, desc="train epoch {}".format(epoch), leave=False)
    amp_enabled = device.startswith("cuda") and not args.no_amp
    for batch in progress:
        images = batch["images"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            image_features = model.encode_image(images)
            outputs = model.predict_from_features(image_features, batch["prompts"])
            negative_outputs = None
            if model.language_conditioning and not args.no_negative_prompts:
                negative_outputs = model.predict_from_features(
                    image_features, batch["negative_prompts"]
                )
            targets = build_grasp_targets(
                batch["grasps"],
                input_size=images.shape[-2:],
                output_size=outputs["center"].shape[-2:],
                device=images.device,
            )
            losses = pcgh_loss(
                outputs,
                targets,
                negative_outputs,
                consistency_weight=args.consistency_weight,
            )
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(losses["total"].detach())
        steps += 1
        progress.set_postfix(loss="{:.4f}".format(running_loss / steps))
        if args.max_steps and steps >= args.max_steps:
            break
    return running_loss / max(1, steps)


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(args.train_manifest, args, shuffle=True)
    val_loader = make_loader(args.val_manifest, args, shuffle=False) if args.val_manifest else None

    start_epoch = 1
    if args.resume:
        model, checkpoint = load_model(args.resume, args.device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
    else:
        model = PCGHNet(
            text_encoder=args.text_encoder,
            pretrained_backbone=not args.no_pretrained_backbone,
            language_conditioning=not args.image_only,
        ).to(args.device)
        checkpoint = {}

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.resume and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - start_epoch + 1)
    )
    amp_enabled = args.device.startswith("cuda") and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, init_scale=1024.0)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable)
    print("parameters: {:,} total, {:,} trainable".format(parameter_count, trainable_count))

    history = list(checkpoint.get("history", []))
    best_score = float(checkpoint.get("best_score", float("-inf")))
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, args.device, args, epoch
        )
        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            metrics.update(evaluate_model(model, val_loader, args.device))
        history.append(metrics)
        scheduler.step()
        print(json.dumps(metrics, sort_keys=True))
        score = metrics.get("success_rate", -train_loss)
        is_best = score > best_score
        if is_best:
            best_score = score
        extra = {"history": history, "best_score": best_score}
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, extra=extra)
        if is_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, extra=extra)
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
