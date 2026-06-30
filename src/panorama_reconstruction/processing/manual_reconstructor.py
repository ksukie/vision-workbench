"""Manual and assisted control-point panorama reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..domain import (
    ControlPointReconstructionParams,
    ImageArray,
    PanoramaResult,
    PathLike,
    Point,
    PointPair,
)
from ..infrastructure import ensure_color_bgr
from .sift_reconstructor import (
    blend_images,
    crop_black_border,
    get_panorama_transform,
)


def normalize_point_pairs(pairs: Iterable[PointPair]) -> List[PointPair]:
    normalized = []
    for left_point, right_point in pairs:
        normalized.append(
            (
                (float(left_point[0]), float(left_point[1])),
                (float(right_point[0]), float(right_point[1])),
            )
        )
    return normalized


def save_point_pairs(path: PathLike, pairs: Iterable[PointPair]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "left": [float(left_point[0]), float(left_point[1])],
            "right": [float(right_point[0]), float(right_point[1])],
        }
        for left_point, right_point in normalize_point_pairs(pairs)
    ]
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_point_pairs(path: PathLike) -> List[PointPair]:
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Point-pair file must contain a JSON list.")

    pairs = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict) or "left" not in item or "right" not in item:
            raise ValueError(f"Point pair {index} must contain left and right.")
        pairs.append((tuple(item["left"]), tuple(item["right"])))
    return normalize_point_pairs(pairs)


def draw_point(image: ImageArray, point: Point, color: Tuple[int, int, int], label: str = "") -> None:
    x, y = np.round(point).astype(int)
    cv2.line(image, (x - 8, y), (x + 8, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (x, y - 8), (x, y + 8), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 5, color, -1)
    cv2.circle(image, (x, y), 9, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(
            image,
            label,
            (x + 11, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def pad_to_height(image: ImageArray, target_height: int) -> ImageArray:
    height = image.shape[0]
    if height >= target_height:
        return image
    return cv2.copyMakeBorder(
        image,
        0,
        target_height - height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def draw_point_pairs(
    left: ImageArray,
    right: ImageArray,
    manual_pairs: Iterable[PointPair],
    assisted_pairs: Iterable[PointPair] = (),
) -> ImageArray:
    left_display = ensure_color_bgr(left).copy()
    right_display = ensure_color_bgr(right).copy()
    manual_pairs = normalize_point_pairs(manual_pairs)
    assisted_pairs = normalize_point_pairs(assisted_pairs)

    for index, (left_point, right_point) in enumerate(manual_pairs, start=1):
        draw_point(left_display, left_point, (0, 255, 255), f"M{index}")
        draw_point(right_display, right_point, (0, 255, 255), f"M{index}")

    for left_point, right_point in assisted_pairs:
        draw_point(left_display, left_point, (0, 255, 0))
        draw_point(right_display, right_point, (0, 0, 255))

    target_height = max(left_display.shape[0], right_display.shape[0])
    left_display = pad_to_height(left_display, target_height)
    right_display = pad_to_height(right_display, target_height)
    canvas = cv2.hconcat([left_display, right_display])

    for left_point, right_point in manual_pairs:
        left_xy = tuple(np.round(left_point).astype(int))
        right_xy = tuple(np.round((right_point[0] + left.shape[1], right_point[1])).astype(int))
        cv2.line(canvas, left_xy, right_xy, (0, 255, 255), 1, cv2.LINE_AA)

    for left_point, right_point in assisted_pairs:
        left_xy = tuple(np.round(left_point).astype(int))
        right_xy = tuple(np.round((right_point[0] + left.shape[1], right_point[1])).astype(int))
        cv2.line(canvas, left_xy, right_xy, (80, 80, 80), 1, cv2.LINE_AA)

    return canvas


def estimate_affine_homography(right_points: ImageArray, left_points: ImageArray) -> ImageArray:
    if len(right_points) == 3:
        affine = cv2.getAffineTransform(right_points[:3], left_points[:3])
    else:
        affine, _ = cv2.estimateAffine2D(
            right_points,
            left_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )

    if affine is None:
        raise ValueError("Could not estimate affine transform from control points.")
    return np.vstack([affine, [0, 0, 1]]).astype(np.float64)


def estimate_perspective_homography(right_points: ImageArray, left_points: ImageArray) -> ImageArray:
    if len(right_points) == 4:
        return cv2.getPerspectiveTransform(right_points[:4], left_points[:4])

    homography, _ = cv2.findHomography(right_points, left_points, cv2.RANSAC, 5.0)
    if homography is None:
        raise ValueError("Could not estimate perspective homography from control points.")
    return homography


def estimate_seed_homography(source_points: Sequence[Point], target_points: Sequence[Point]) -> ImageArray:
    source = np.float32(source_points)
    target = np.float32(target_points)
    if len(source) < 3:
        raise ValueError("Need at least 3 point pairs for a seed transform.")

    if len(source) == 3:
        affine = cv2.getAffineTransform(source[:3], target[:3])
        return np.vstack([affine, [0, 0, 1]]).astype(np.float64)

    homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    if homography is not None:
        return homography.astype(np.float64)

    affine, _ = cv2.estimateAffine2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if affine is None:
        raise ValueError("Could not estimate a seed transform from control points.")
    return np.vstack([affine, [0, 0, 1]]).astype(np.float64)


def transform_point(point: Point, homography: ImageArray) -> Point:
    point_array = np.float32([[point]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(point_array, homography)
    x, y = transformed.reshape(2)
    return float(x), float(y)


def patch_inside(image_shape: Tuple[int, ...], point: Point, radius: int) -> bool:
    height, width = image_shape[:2]
    x, y = point
    return radius <= x < width - radius and radius <= y < height - radius


def match_template_near(
    left_gray: ImageArray,
    right_gray: ImageArray,
    left_point: Point,
    predicted_right_point: Point,
    template_radius: int,
    search_radius: int,
    min_score: float,
    min_std: float,
) -> Optional[Tuple[float, float, float]]:
    if not patch_inside(left_gray.shape, left_point, template_radius):
        return None

    left_x, left_y = np.round(left_point).astype(int)
    pred_x, pred_y = np.round(predicted_right_point).astype(int)
    search_pad = search_radius + template_radius
    if not patch_inside(right_gray.shape, (float(pred_x), float(pred_y)), search_pad):
        return None

    template = left_gray[
        left_y - template_radius : left_y + template_radius + 1,
        left_x - template_radius : left_x + template_radius + 1,
    ]
    if float(template.std()) < min_std:
        return None

    search = right_gray[
        pred_y - search_pad : pred_y + search_pad + 1,
        pred_x - search_pad : pred_x + search_pad + 1,
    ]
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_location = cv2.minMaxLoc(scores)
    if max_score < min_score:
        return None

    matched_x = pred_x - search_pad + max_location[0] + template_radius
    matched_y = pred_y - search_pad + max_location[1] + template_radius
    return float(matched_x), float(matched_y), float(max_score)


def point_far_enough(point: Point, accepted_points: Sequence[Point], min_distance: float) -> bool:
    if not accepted_points:
        return True
    point_array = np.float32(point)
    accepted = np.float32(accepted_points)
    distances = np.linalg.norm(accepted - point_array, axis=1)
    return bool(np.all(distances >= min_distance))


def generate_guided_matches(
    left: ImageArray,
    right: ImageArray,
    manual_pairs: Sequence[PointPair],
    params: ControlPointReconstructionParams,
) -> List[PointPair]:
    manual_pairs = normalize_point_pairs(manual_pairs)
    left_points = [left_point for left_point, _ in manual_pairs]
    right_points = [right_point for _, right_point in manual_pairs]
    left_to_right = estimate_seed_homography(left_points, right_points)

    left_gray = cv2.cvtColor(ensure_color_bgr(left), cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(ensure_color_bgr(right), cv2.COLOR_BGR2GRAY)
    height, width = left_gray.shape[:2]
    candidates = []

    for y in range(params.auto_template_radius, height - params.auto_template_radius, params.auto_grid_step):
        for x in range(params.auto_template_radius, width - params.auto_template_radius, params.auto_grid_step):
            left_point = (float(x), float(y))
            predicted = transform_point(left_point, left_to_right)
            match = match_template_near(
                left_gray,
                right_gray,
                left_point,
                predicted,
                params.auto_template_radius,
                params.auto_search_radius,
                params.auto_min_score,
                params.auto_min_std,
            )
            if match is None:
                continue

            matched_x, matched_y, score = match
            candidates.append((score, left_point, (matched_x, matched_y)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    assisted_pairs = []
    accepted_left_points = list(left_points)
    min_distance = max(float(params.auto_grid_step) * 0.65, float(params.auto_template_radius) * 2.0)

    for _score, left_point, right_point in candidates:
        if not point_far_enough(left_point, accepted_left_points, min_distance):
            continue
        assisted_pairs.append((left_point, right_point))
        accepted_left_points.append(left_point)
        if len(assisted_pairs) >= params.auto_max_points:
            break

    return normalize_point_pairs(assisted_pairs)


def remove_near_duplicate_pairs(pairs: Iterable[PointPair], min_distance: float = 1.0) -> List[PointPair]:
    kept_pairs = []
    kept_left_points = []
    for left_point, right_point in normalize_point_pairs(pairs):
        if point_far_enough(left_point, kept_left_points, min_distance):
            kept_pairs.append((left_point, right_point))
            kept_left_points.append(left_point)
    return kept_pairs


def warp_right_to_left(left: ImageArray, right: ImageArray, right_to_left_homography: ImageArray) -> Tuple[ImageArray, ImageArray]:
    translation, panorama_size, left_offset = get_panorama_transform(left, right, right_to_left_homography)
    warped_right = cv2.warpPerspective(right, translation @ right_to_left_homography, panorama_size)

    left_canvas = np.zeros_like(warped_right)
    offset_x, offset_y = left_offset
    height, width = left.shape[:2]
    left_canvas[offset_y : offset_y + height, offset_x : offset_x + width] = left

    panorama = blend_images(left_canvas, warped_right)
    panorama = crop_black_border(panorama)
    return panorama, warped_right


def draw_mapped_control_points_on_left(
    left: ImageArray,
    pairs: Sequence[PointPair],
    homography: ImageArray,
) -> ImageArray:
    pairs = normalize_point_pairs(pairs)
    left_points = np.float32([left_point for left_point, _ in pairs]).reshape(-1, 1, 2)
    right_points = np.float32([right_point for _, right_point in pairs]).reshape(-1, 1, 2)
    mapped_right_points = cv2.perspectiveTransform(right_points, homography)

    mapped_image = ensure_color_bgr(left).copy()
    for index, (left_point, mapped_point) in enumerate(
        zip(left_points.reshape(-1, 2), mapped_right_points.reshape(-1, 2)),
        start=1,
    ):
        left_xy = tuple(np.round(left_point).astype(int))
        mapped_xy = tuple(np.round(mapped_point).astype(int))
        cv2.circle(mapped_image, left_xy, 6, (0, 255, 0), 2)
        cv2.circle(mapped_image, mapped_xy, 4, (0, 0, 255), -1)
        cv2.line(mapped_image, left_xy, mapped_xy, (255, 0, 0), 1)
        cv2.putText(
            mapped_image,
            str(index),
            (left_xy[0] + 8, left_xy[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return mapped_image


def reconstruct_manual(
    left: ImageArray,
    right: ImageArray,
    point_pairs: Sequence[PointPair],
) -> PanoramaResult:
    left = ensure_color_bgr(left)
    right = ensure_color_bgr(right)
    pairs = normalize_point_pairs(point_pairs)
    if len(pairs) < 3:
        raise ValueError("Manual reconstruction needs at least 3 point pairs.")

    left_points = np.float32([left_point for left_point, _ in pairs])
    right_points = np.float32([right_point for _, right_point in pairs])
    if len(pairs) >= 4:
        homography = estimate_perspective_homography(right_points, left_points)
        method = "manual-perspective"
    else:
        homography = estimate_affine_homography(right_points, left_points)
        method = "manual-affine"

    panorama, warped_right = warp_right_to_left(left, right, homography)
    return PanoramaResult(
        panorama=panorama,
        warped_right=warped_right,
        match_visualization=draw_point_pairs(left, right, pairs),
        mapped_points_visualization=draw_mapped_control_points_on_left(left, pairs, homography),
        raw_match_count=len(pairs),
        balanced_match_count=len(pairs),
        inlier_count=len(pairs),
        channel_name="control-points",
        method=method,
        extra_metrics={"manual_pairs": len(pairs), "assisted_pairs": 0},
    )


def tps_kernel_from_squared_distance(squared_distance: ImageArray) -> ImageArray:
    kernel = np.zeros_like(squared_distance, dtype=np.float64)
    mask = squared_distance > 0
    kernel[mask] = squared_distance[mask] * np.log(squared_distance[mask])
    return kernel


def fit_tps_map(source_points: ImageArray, target_points: ImageArray, smooth: float, coordinate_scale: float) -> dict:
    source = np.float64(source_points) / coordinate_scale
    target = np.float64(target_points)
    point_count = len(source)
    if point_count < 3:
        raise ValueError("TPS needs at least 3 point pairs.")

    deltas = source[:, None, :] - source[None, :, :]
    squared_distance = np.sum(deltas * deltas, axis=2)
    kernel = tps_kernel_from_squared_distance(squared_distance)
    if smooth > 0:
        kernel += np.eye(point_count, dtype=np.float64) * float(smooth)

    polynomial = np.column_stack([np.ones(point_count, dtype=np.float64), source[:, 0], source[:, 1]])
    system = np.zeros((point_count + 3, point_count + 3), dtype=np.float64)
    system[:point_count, :point_count] = kernel
    system[:point_count, point_count:] = polynomial
    system[point_count:, :point_count] = polynomial.T

    values = np.zeros((point_count + 3, 2), dtype=np.float64)
    values[:point_count] = target

    try:
        coefficients = np.linalg.solve(system, values)
    except np.linalg.LinAlgError:
        coefficients, *_ = np.linalg.lstsq(system, values, rcond=None)

    return {
        "points": source,
        "coefficients": coefficients,
        "coordinate_scale": float(coordinate_scale),
    }


def evaluate_tps(model: dict, query_points: ImageArray) -> ImageArray:
    query = np.float64(query_points) / model["coordinate_scale"]
    source = model["points"]
    coefficients = model["coefficients"]
    point_count = len(source)

    deltas = query[:, None, :] - source[None, :, :]
    squared_distance = np.sum(deltas * deltas, axis=2)
    kernel = tps_kernel_from_squared_distance(squared_distance)
    polynomial = np.column_stack([np.ones(len(query), dtype=np.float64), query[:, 0], query[:, 1]])
    basis = np.hstack([kernel, polynomial])
    return basis @ coefficients[: point_count + 3]


def make_tps_remap(model: dict, output_shape: Tuple[int, int], origin: Point, chunk_rows: int = 24) -> Tuple[ImageArray, ImageArray]:
    height, width = output_shape[:2]
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    origin_x, origin_y = origin
    x_values = np.arange(width, dtype=np.float64) + float(origin_x)

    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        y_values = np.arange(y0, y1, dtype=np.float64) + float(origin_y)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        query = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        mapped = evaluate_tps(model, query)
        map_x[y0:y1] = mapped[:, 0].reshape(y1 - y0, width).astype(np.float32)
        map_y[y0:y1] = mapped[:, 1].reshape(y1 - y0, width).astype(np.float32)

    return map_x, map_y


def sample_image_boundary_points(image_shape: Tuple[int, ...], step: int) -> ImageArray:
    height, width = image_shape[:2]
    step = max(1, int(step))
    xs = np.unique(np.r_[np.arange(0, width, step), width - 1]).astype(np.float64)
    ys = np.unique(np.r_[np.arange(0, height, step), height - 1]).astype(np.float64)

    top = np.column_stack([xs, np.zeros_like(xs)])
    bottom = np.column_stack([xs, np.full_like(xs, height - 1)])
    left_edge = np.column_stack([np.zeros_like(ys), ys])
    right_edge = np.column_stack([np.full_like(ys, width - 1), ys])
    return np.vstack([top, bottom, left_edge, right_edge])


def get_tps_canvas(
    left: ImageArray,
    right: ImageArray,
    left_points: ImageArray,
    right_points: ImageArray,
    smooth: float,
    coordinate_scale: float,
    padding: int,
    boundary_step: int,
) -> Tuple[Point, Tuple[int, int]]:
    inverse_model = fit_tps_map(right_points, left_points, smooth, coordinate_scale)
    right_boundary = sample_image_boundary_points(right.shape[:2], boundary_step)
    mapped_right_boundary = evaluate_tps(inverse_model, right_boundary)
    left_boundary = sample_image_boundary_points(left.shape[:2], boundary_step)
    all_points = np.vstack([left_boundary, mapped_right_boundary, left_points])

    min_x, min_y = np.floor(all_points.min(axis=0) - padding).astype(int)
    max_x, max_y = np.ceil(all_points.max(axis=0) + padding).astype(int)
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)

    max_reasonable_width = (left.shape[1] + right.shape[1]) * 4
    max_reasonable_height = max(left.shape[0], right.shape[0]) * 4
    if (
        width <= 0
        or height <= 0
        or width > max_reasonable_width
        or height > max_reasonable_height
    ):
        return (0.0, 0.0), (left.shape[1], left.shape[0])

    return (float(min_x), float(min_y)), (width, height)


def paste_left_on_canvas(left: ImageArray, canvas_size: Tuple[int, int], origin: Point) -> Tuple[ImageArray, ImageArray]:
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=left.dtype)
    origin_x, origin_y = origin
    offset_x = -int(origin_x)
    offset_y = -int(origin_y)
    height, width = left.shape[:2]
    canvas[offset_y : offset_y + height, offset_x : offset_x + width] = left
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    mask[offset_y : offset_y + height, offset_x : offset_x + width] = 255
    return canvas, mask


def restore_with_tps(
    left: ImageArray,
    right: ImageArray,
    pairs: Sequence[PointPair],
    params: ControlPointReconstructionParams,
) -> Tuple[ImageArray, ImageArray, ImageArray]:
    pairs = remove_near_duplicate_pairs(pairs)
    left_points = np.float64([left_point for left_point, _ in pairs])
    right_points = np.float64([right_point for _, right_point in pairs])
    coordinate_scale = float(max(left.shape[0], left.shape[1], right.shape[0], right.shape[1]))
    model = fit_tps_map(left_points, right_points, params.tps_smooth, coordinate_scale)
    origin, panorama_size = get_tps_canvas(
        left,
        right,
        left_points,
        right_points,
        params.tps_smooth,
        coordinate_scale,
        params.tps_padding,
        params.tps_boundary_step,
    )
    panorama_width, panorama_height = panorama_size
    map_x, map_y = make_tps_remap(model, (panorama_height, panorama_width), origin=origin)

    warped_right = cv2.remap(
        right,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    source_mask = np.full(right.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.remap(
        source_mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    left_canvas, left_mask = paste_left_on_canvas(left, panorama_size, origin)
    panorama = left_canvas.copy()
    overlay_only = warped_mask > 0
    overlap = overlay_only & (left_mask > 0)
    panorama[overlay_only] = warped_right[overlay_only]
    panorama[overlap] = cv2.addWeighted(left_canvas, 0.5, warped_right, 0.5, 0)[overlap]
    return crop_black_border(panorama), warped_right, warped_mask


def reconstruct_manual_assisted(
    left: ImageArray,
    right: ImageArray,
    point_pairs: Sequence[PointPair],
    params: ControlPointReconstructionParams = ControlPointReconstructionParams(),
) -> PanoramaResult:
    left = ensure_color_bgr(left)
    right = ensure_color_bgr(right)
    manual_pairs = normalize_point_pairs(point_pairs)
    if len(manual_pairs) < 3:
        raise ValueError("Assisted reconstruction needs at least 3 manual point pairs.")

    assisted_pairs = generate_guided_matches(left, right, manual_pairs, params)
    all_pairs = remove_near_duplicate_pairs([*manual_pairs, *assisted_pairs])
    if len(all_pairs) < 3:
        raise ValueError("Assisted reconstruction did not produce enough point pairs.")

    panorama, warped_right, _warped_mask = restore_with_tps(left, right, all_pairs, params)
    left_points = np.float32([left_point for left_point, _ in all_pairs])
    right_points = np.float32([right_point for _, right_point in all_pairs])
    homography = estimate_seed_homography(right_points, left_points)

    return PanoramaResult(
        panorama=panorama,
        warped_right=warped_right,
        match_visualization=draw_point_pairs(left, right, manual_pairs, assisted_pairs),
        mapped_points_visualization=draw_mapped_control_points_on_left(left, all_pairs, homography),
        raw_match_count=len(all_pairs),
        balanced_match_count=len(all_pairs),
        inlier_count=len(all_pairs),
        channel_name="control-points",
        method="manual-assisted-tps",
        extra_metrics={
            "manual_pairs": len(manual_pairs),
            "assisted_pairs": len(assisted_pairs),
            "total_pairs": len(all_pairs),
        },
    )
