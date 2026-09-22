import math

import torch


def decode_grasps(outputs, input_size, top_k=1, score_threshold=0.0):
    """Decode dense model outputs into [x, y, width, height, theta, score]."""
    scores = outputs["center"].sigmoid()
    batch_size, _, out_h, out_w = scores.shape
    in_h, in_w = input_size
    count = min(top_k, out_h * out_w)
    values, indices = scores.flatten(1).topk(count, dim=1)
    decoded = []
    for batch_index in range(batch_size):
        grasps = []
        for score, flat_index in zip(values[batch_index], indices[batch_index]):
            if float(score) < score_threshold:
                continue
            y = int(flat_index) // out_w
            x = int(flat_index) % out_w
            offset_x = float(outputs["offset"][batch_index, 0, y, x])
            offset_y = float(outputs["offset"][batch_index, 1, y, x])
            width = float(outputs["size"][batch_index, 0, y, x]) * in_w
            height = float(outputs["size"][batch_index, 1, y, x]) * in_h
            sin2 = float(outputs["angle"][batch_index, 0, y, x])
            cos2 = float(outputs["angle"][batch_index, 1, y, x])
            theta = 0.5 * math.atan2(sin2, cos2)
            grasps.append(
                [
                    (x + offset_x) / out_w * in_w,
                    (y + offset_y) / out_h * in_h,
                    width,
                    height,
                    theta,
                    float(score),
                ]
            )
        decoded.append(grasps)
    return decoded
