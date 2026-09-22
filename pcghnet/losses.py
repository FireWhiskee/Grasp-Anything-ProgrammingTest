import torch
import torch.nn.functional as F


def focal_heatmap_loss(logits, target, alpha=2.0, beta=4.0):
    prediction = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    positive = target.eq(1.0).to(prediction.dtype)
    negative = target.lt(1.0).to(prediction.dtype)
    negative_weight = (1.0 - target).pow(beta)
    positive_loss = -(prediction.log()) * (1.0 - prediction).pow(alpha) * positive
    negative_loss = -(1.0 - prediction).log() * prediction.pow(alpha) * negative_weight * negative
    positive_count = positive.sum().clamp_min(1.0)
    return (positive_loss.sum() + negative_loss.sum()) / positive_count


def masked_smooth_l1(prediction, target, mask):
    expanded_mask = mask.expand_as(prediction)
    denominator = expanded_mask.sum().clamp_min(1.0)
    return (F.smooth_l1_loss(prediction, target, reduction="none") * expanded_mask).sum() / denominator


def masked_angle_loss(prediction, target, mask):
    prediction = F.normalize(prediction, dim=1, eps=1e-6)
    similarity = (prediction * target).sum(dim=1, keepdim=True)
    return ((1.0 - similarity) * mask).sum() / mask.sum().clamp_min(1.0)


def negative_prompt_consistency_loss(positive_logits, negative_logits, target_heatmap, margin=0.2):
    """Require the correct prompt to score the target region above a wrong prompt."""
    region = target_heatmap.gt(0.3).to(positive_logits.dtype)
    region_count = region.flatten(1).sum(1).clamp_min(1.0)
    positive_score = (positive_logits.sigmoid() * region).flatten(1).sum(1) / region_count
    negative_score = (negative_logits.sigmoid() * region).flatten(1).sum(1) / region_count
    return F.relu(margin - positive_score + negative_score).mean()


def pcgh_loss(outputs, targets, negative_outputs=None, consistency_weight=0.25):
    losses = {
        "center": focal_heatmap_loss(outputs["center"], targets["center"]),
        "offset": masked_smooth_l1(
            outputs["offset"], targets["offset"], targets["regression_mask"]
        ),
        "size": masked_smooth_l1(outputs["size"], targets["size"], targets["regression_mask"]),
        "angle": masked_angle_loss(outputs["angle"], targets["angle"], targets["regression_mask"]),
    }
    if negative_outputs is not None:
        losses["consistency"] = negative_prompt_consistency_loss(
            outputs["center"], negative_outputs["center"], targets["center"]
        )
    else:
        losses["consistency"] = outputs["center"].new_zeros(())
    losses["total"] = (
        losses["center"]
        + losses["offset"]
        + losses["size"]
        + losses["angle"]
        + consistency_weight * losses["consistency"]
    )
    return losses
