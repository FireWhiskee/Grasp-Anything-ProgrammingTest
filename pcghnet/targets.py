import math

import torch


def _gaussian_radius(width, height, min_overlap=0.7):
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    radius = (b1 - math.sqrt(max(0.0, b1 * b1 - 4.0 * a1 * c1))) / 2.0
    return max(1, int(radius))


def _draw_gaussian(heatmap, center_x, center_y, radius):
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    coordinates = torch.arange(diameter, device=heatmap.device, dtype=heatmap.dtype) - radius
    gaussian = torch.exp(
        -(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2.0 * sigma * sigma)
    )
    height, width = heatmap.shape
    left, right = min(center_x, radius), min(width - center_x - 1, radius)
    top, bottom = min(center_y, radius), min(height - center_y - 1, radius)
    heatmap_slice = heatmap[center_y - top : center_y + bottom + 1, center_x - left : center_x + right + 1]
    gaussian_slice = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    torch.maximum(heatmap_slice, gaussian_slice, out=heatmap_slice)


def build_grasp_targets(batch_grasps, input_size, output_size, device=None):
    """Rasterize lists of [x, y, width, height, theta_radians] grasps."""
    batch_size = len(batch_grasps)
    out_h, out_w = output_size
    in_h, in_w = input_size
    center = torch.zeros(batch_size, 1, out_h, out_w, device=device)
    offset = torch.zeros(batch_size, 2, out_h, out_w, device=device)
    size = torch.zeros(batch_size, 2, out_h, out_w, device=device)
    angle = torch.zeros(batch_size, 2, out_h, out_w, device=device)
    regression_mask = torch.zeros(batch_size, 1, out_h, out_w, device=device)

    for batch_index, grasps in enumerate(batch_grasps):
        for grasp in grasps:
            x, y, width, height, theta = [float(value) for value in grasp]
            exact_x = min(out_w - 1e-4, max(0.0, x / in_w * out_w))
            exact_y = min(out_h - 1e-4, max(0.0, y / in_h * out_h))
            center_x = int(exact_x)
            center_y = int(exact_y)
            scaled_width = max(1.0, width / in_w * out_w)
            scaled_height = max(1.0, height / in_h * out_h)
            radius = _gaussian_radius(scaled_width, scaled_height)
            _draw_gaussian(center[batch_index, 0], center_x, center_y, radius)
            offset[batch_index, :, center_y, center_x] = torch.tensor(
                [exact_x - center_x, exact_y - center_y], device=device
            )
            size[batch_index, :, center_y, center_x] = torch.tensor(
                [width / in_w, height / in_h], device=device
            )
            angle[batch_index, :, center_y, center_x] = torch.tensor(
                [math.sin(2.0 * theta), math.cos(2.0 * theta)], device=device
            )
            regression_mask[batch_index, 0, center_y, center_x] = 1.0

    return {
        "center": center,
        "offset": offset,
        "size": size,
        "angle": angle,
        "regression_mask": regression_mask,
    }
