"""Geometry utilities for evaluating antipodal rectangular grasps."""

import math


def normalize_grasp_angle(theta):
    """Map an antipodal grasp angle to [-pi/2, pi/2)."""
    return (float(theta) + math.pi / 2.0) % math.pi - math.pi / 2.0


def grasp_angle_error(first, second):
    return abs(normalize_grasp_angle(float(first) - float(second)))


def rectangle_corners(grasp):
    x, y, width, height, theta = [float(value) for value in grasp[:5]]
    half_width = width / 2.0
    half_height = height / 2.0
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    along = (cos_theta, sin_theta)
    across = (-sin_theta, cos_theta)
    corners = []
    for along_scale, across_scale in (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ):
        corners.append(
            (
                x + along_scale * along[0] + across_scale * across[0],
                y + along_scale * along[1] + across_scale * across[1],
            )
        )
    return corners


def _cross(first, second):
    return first[0] * second[1] - first[1] * second[0]


def _subtract(first, second):
    return first[0] - second[0], first[1] - second[1]


def _inside(point, edge_start, edge_end):
    return _cross(_subtract(edge_end, edge_start), _subtract(point, edge_start)) >= -1e-8


def _line_intersection(segment_start, segment_end, edge_start, edge_end):
    segment = _subtract(segment_end, segment_start)
    edge = _subtract(edge_end, edge_start)
    denominator = _cross(segment, edge)
    if abs(denominator) < 1e-12:
        return segment_end
    distance = _cross(_subtract(edge_start, segment_start), edge) / denominator
    return segment_start[0] + distance * segment[0], segment_start[1] + distance * segment[1]


def _clip_polygon(subject, clipper):
    output = list(subject)
    for index, edge_start in enumerate(clipper):
        edge_end = clipper[(index + 1) % len(clipper)]
        input_polygon = output
        output = []
        if not input_polygon:
            break
        previous = input_polygon[-1]
        for current in input_polygon:
            current_inside = _inside(current, edge_start, edge_end)
            previous_inside = _inside(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, edge_start, edge_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def _polygon_area(points):
    if len(points) < 3:
        return 0.0
    twice_area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        twice_area += point[0] * next_point[1] - point[1] * next_point[0]
    return abs(twice_area) / 2.0


def rotated_iou(first, second):
    first_polygon = rectangle_corners(first)
    second_polygon = rectangle_corners(second)
    intersection = _polygon_area(_clip_polygon(first_polygon, second_polygon))
    union = _polygon_area(first_polygon) + _polygon_area(second_polygon) - intersection
    return intersection / union if union > 1e-12 else 0.0


def best_grasp_match(prediction, ground_truth):
    """Return the GT match with highest IoU and its antipodal angle error."""
    best_iou = 0.0
    best_angle_error = math.pi / 2.0
    for grasp in ground_truth:
        overlap = rotated_iou(prediction, grasp)
        error = grasp_angle_error(prediction[4], grasp[4])
        if overlap > best_iou:
            best_iou = overlap
            best_angle_error = error
    return best_iou, best_angle_error


def is_successful_grasp(
    prediction, ground_truth, iou_threshold=0.25, angle_threshold_degrees=30.0
):
    overlap, angle_error = best_grasp_match(prediction, ground_truth)
    success = overlap >= iou_threshold and angle_error <= math.radians(angle_threshold_degrees)
    return success, overlap, angle_error
