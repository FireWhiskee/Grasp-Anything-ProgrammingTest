from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

try:
    from torchvision.models import ResNet18_Weights
except ImportError:  # torchvision < 0.13
    ResNet18_Weights = None

from .text_encoder import build_text_encoder


class TextFiLM(nn.Module):
    """Feature-wise affine modulation conditioned on a prompt embedding."""

    def __init__(self, text_dim, channels):
        super().__init__()
        self.affine = nn.Linear(text_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, feature, text_embedding):
        gamma, beta = self.affine(text_embedding).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return feature * (1.0 + gamma) + beta


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PCGHNet(nn.Module):
    """Prompt-conditioned dense predictor for 2D rectangular grasps.

    The output resolution is one quarter of the input resolution. Width and
    height are normalized to [0, 1], while orientation is represented by
    sin(2*theta) and cos(2*theta) to respect antipodal grasp symmetry.
    """

    def __init__(
        self,
        text_encoder="hash",
        text_model_name=None,
        text_dim=256,
        fpn_dim=128,
        pretrained_backbone=False,
    ):
        super().__init__()
        self.model_config = {
            "text_encoder": text_encoder,
            "text_model_name": text_model_name,
            "text_dim": text_dim,
            "fpn_dim": fpn_dim,
            "pretrained_backbone": pretrained_backbone,
        }
        self.text_encoder = build_text_encoder(text_encoder, text_dim, text_model_name)

        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(pretrained=pretrained_backbone)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stages = nn.ModuleList(
            [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
        )
        channels = [64, 128, 256, 512]
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in channels])
        self.film = nn.ModuleList([TextFiLM(text_dim, fpn_dim) for _ in channels])
        self.smooth = nn.ModuleList([ConvBlock(fpn_dim, fpn_dim) for _ in channels])

        self.fusion = ConvBlock(fpn_dim * 4, fpn_dim)
        self.heads = nn.ModuleDict(
            OrderedDict(
                center=nn.Conv2d(fpn_dim, 1, 1),
                offset=nn.Conv2d(fpn_dim, 2, 1),
                size=nn.Conv2d(fpn_dim, 2, 1),
                angle=nn.Conv2d(fpn_dim, 2, 1),
            )
        )
        nn.init.constant_(self.heads["center"].bias, -2.19)

    def encode_image(self, images):
        features = []
        feature = self.stem(images)
        for stage in self.stages:
            feature = stage(feature)
            features.append(feature)
        return features

    def predict_from_features(self, features, prompts):
        if features[0].shape[0] != len(prompts):
            raise ValueError("Batch size and number of prompts must match")
        text_embedding = self.text_encoder(prompts)

        pyramid = [None] * len(features)
        top_down = None
        for index in range(len(features) - 1, -1, -1):
            current = self.lateral[index](features[index])
            if top_down is not None:
                current = current + F.interpolate(
                    top_down, size=current.shape[-2:], mode="bilinear", align_corners=False
                )
            current = self.film[index](current, text_embedding)
            pyramid[index] = self.smooth[index](current)
            top_down = current

        target_size = pyramid[0].shape[-2:]
        fused = torch.cat(
            [
                level
                if level.shape[-2:] == target_size
                else F.interpolate(level, size=target_size, mode="bilinear", align_corners=False)
                for level in pyramid
            ],
            dim=1,
        )
        fused = self.fusion(fused)
        return {
            "center": self.heads["center"](fused),
            "offset": torch.sigmoid(self.heads["offset"](fused)),
            "size": torch.sigmoid(self.heads["size"](fused)),
            "angle": self.heads["angle"](fused),
        }

    def forward(self, images, prompts):
        return self.predict_from_features(self.encode_image(images), prompts)
