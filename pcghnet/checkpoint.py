import torch

from .model import PCGHNet


def save_checkpoint(path, model, optimizer=None, epoch=0):
    payload = {
        "model": model.state_dict(),
        "model_config": model.model_config,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = PCGHNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint
