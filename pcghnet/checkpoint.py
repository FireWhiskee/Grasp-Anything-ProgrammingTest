import torch

from .model import PCGHNet


def save_checkpoint(path, model, optimizer=None, epoch=0, extra=None):
    payload = {
        "model": model.state_dict(),
        "model_config": model.model_config,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model_config = dict(checkpoint["model_config"])
    # State-dict loading replaces every backbone parameter, so avoid a needless
    # network download when restoring a checkpoint trained with ImageNet init.
    model_config["pretrained_backbone"] = False
    model = PCGHNet(**model_config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint
