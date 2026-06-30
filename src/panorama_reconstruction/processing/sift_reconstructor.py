"""SIFT-based right-to-left panorama reconstruction."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from ..configuration import CHANNEL_CHOICES
from ..domain import ImageArray, PanoramaReconstructionParams, PanoramaResult
from ..infrastructure import ensure_color_bgr


def resize_keep_ratio(image: ImageArray, target_width: int) -> ImageArray:
    height, width = image.shape[:2]
    target_height = max(1, int(height * target_width / width))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


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


def channel_to_bgr(channel: ImageArray) -> ImageArray:
    normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def add_label(image: ImageArray, label: str) -> ImageArray:
    label_height = 36
    labeled = cv2.copyMakeBorder(
        image,
        label_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    cv2.putText(
        labeled,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def add_header(image: ImageArray, title: str) -> ImageArray:
    header_height = 42
    titled = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    font_scale = 0.72
    thickness = 2
    max_text_width = image.shape[1] - 20
    while font_scale > 0.35:
        text_width = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if text_width <= max_text_width:
            break
        font_scale -= 0.05

    cv2.putText(
        titled,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return titled


def get_rgb_hsv_channels(image: ImageArray) -> List[Tuple[str, ImageArray]]:
    color = ensure_color_bgr(image)
    blue, green, red = cv2.split(color)
    hue, saturation, value = cv2.split(cv2.cvtColor(color, cv2.COLOR_BGR2HSV))
    return [
        ("R", red),
        ("G", green),
        ("B", blue),
        ("H", hue),
        ("S", saturation),
        ("V", value),
    ]


def get_preview_channels(image: ImageArray) -> List[Tuple[str, ImageArray]]:
    gray = cv2.cvtColor(ensure_color_bgr(image), cv2.COLOR_BGR2GRAY)
    return [("Gray", gray)] + get_rgb_hsv_channels(image)


def normalize_channel_name(channel_name: str) -> str:
    normalized = channel_name.strip().lower()
    if normalized == "grey":
        normalized = "gray"
    if normalized not in CHANNEL_CHOICES:
        choices = ", ".join(CHANNEL_CHOICES)
        raise ValueError(f"Unknown channel: {channel_name}. Choose one of: {choices}")
    return normalized


def get_feature_channel(image: ImageArray, channel_name: str) -> ImageArray:
    channel_name = normalize_channel_name(channel_name)
    color = ensure_color_bgr(image)
    if channel_name == "gray":
        return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    channels = {name.lower(): channel for name, channel in get_rgb_hsv_channels(color)}
    return channels[channel_name]


def create_channel_preview(left: ImageArray, right: ImageArray) -> ImageArray:
    rows = []
    for row_name, image in [("Left", left), ("Right", right)]:
        cells = []
        for channel_name, channel in get_preview_channels(image):
            cell = channel_to_bgr(channel)
            cell = resize_keep_ratio(cell, 220)
            cell = add_label(cell, f"{row_name} {channel_name}")
            cells.append(cell)
        rows.append(cv2.hconcat(cells))

    max_height = max(row.shape[0] for row in rows)
    rows = [pad_to_height(row, max_height) for row in rows]
    return cv2.vconcat(rows)


def create_sift_detector() -> cv2.SIFT:
    common_args = dict(
        nfeatures=0,
        nOctaveLayers=6,
        contrastThreshold=0.03,
        edgeThreshold=8,
        sigma=1.6,
    )
    try:
        return cv2.SIFT_create(**common_args, enable_precise_upscale=True)
    except TypeError:
        return cv2.SIFT_create(**common_args)


def detect_sift_features(
    image: ImageArray,
    channel_name: str = "gray",
) -> Tuple[Sequence[cv2.KeyPoint], ImageArray]:
    channel = get_feature_channel(image, channel_name)
    sift = create_sift_detector()
    keypoints, descriptors = sift.detectAndCompute(channel, None)

    if descriptors is None or len(keypoints) < 4:
        raise ValueError("SIFT could not find enough features in one of the images.")

    return keypoints, descriptors


def match_features(
    left_descriptors: ImageArray,
    right_descriptors: ImageArray,
    ratio_threshold: float,
) -> List[cv2.DMatch]:
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50),
    )
    raw_matches = matcher.knnMatch(right_descriptors, left_descriptors, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue

        first, second = pair
        if first.distance < ratio_threshold * second.distance:
            good_matches.append(first)

    if len(good_matches) < 4:
        raise ValueError(f"Only found {len(good_matches)} good matches. Need at least 4.")

    return good_matches


def balance_matches_by_grid(
    matches: Iterable[cv2.DMatch],
    right_keypoints: Sequence[cv2.KeyPoint],
    image_shape: Tuple[int, ...],
    params: PanoramaReconstructionParams,
) -> List[cv2.DMatch]:
    height, width = image_shape[:2]
    grid = {}

    for match in matches:
        x, y = right_keypoints[match.queryIdx].pt
        col = min(params.match_grid_cols - 1, int(x / width * params.match_grid_cols))
        row = min(params.match_grid_rows - 1, int(y / height * params.match_grid_rows))
        grid.setdefault((row, col), []).append(match)

    balanced_matches = []
    for cell_matches in grid.values():
        cell_matches = sorted(cell_matches, key=lambda item: item.distance)
        balanced_matches.extend(cell_matches[: params.max_matches_per_grid_cell])

    balanced_matches = sorted(balanced_matches, key=lambda item: item.distance)
    if len(balanced_matches) < 4:
        return list(matches)

    return balanced_matches


def estimate_right_to_left_homography(
    left_keypoints: Sequence[cv2.KeyPoint],
    right_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    params: PanoramaReconstructionParams,
) -> Tuple[ImageArray, ImageArray]:
    right_points = np.float32(
        [right_keypoints[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    left_points = np.float32(
        [left_keypoints[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        right_points,
        left_points,
        cv2.RANSAC,
        params.ransac_reprojection_threshold,
    )
    if homography is None:
        raise ValueError("Could not compute homography from SIFT matches.")

    return homography, mask


def get_image_corners(image: ImageArray) -> ImageArray:
    height, width = image.shape[:2]
    return np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
        -1, 1, 2
    )


def get_panorama_transform(
    left: ImageArray,
    right: ImageArray,
    right_to_left_homography: ImageArray,
) -> Tuple[ImageArray, Tuple[int, int], Tuple[int, int]]:
    left_corners = get_image_corners(left)
    right_corners = cv2.perspectiveTransform(
        get_image_corners(right), right_to_left_homography
    )

    all_corners = np.concatenate((left_corners, right_corners), axis=0)
    min_x, min_y = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    translate_x = -min(0, min_x)
    translate_y = -min(0, min_y)
    panorama_width = max_x + translate_x
    panorama_height = max_y + translate_y

    max_reasonable_width = (left.shape[1] + right.shape[1]) * 4
    max_reasonable_height = max(left.shape[0], right.shape[0]) * 4
    if (
        panorama_width <= 0
        or panorama_height <= 0
        or panorama_width > max_reasonable_width
        or panorama_height > max_reasonable_height
    ):
        raise ValueError(
            "Computed panorama canvas is too large. "
            "The selected channel probably produced unreliable matches."
        )

    translation = np.array(
        [[1, 0, translate_x], [0, 1, translate_y], [0, 0, 1]],
        dtype=np.float64,
    )
    return translation, (panorama_width, panorama_height), (translate_x, translate_y)


def blend_images(base: ImageArray, overlay: ImageArray) -> ImageArray:
    base_mask = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) > 0
    overlay_mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0

    panorama = base.copy()
    overlay_only = overlay_mask & ~base_mask
    overlap = base_mask & overlay_mask

    panorama[overlay_only] = overlay[overlay_only]
    panorama[overlap] = cv2.addWeighted(base, 0.5, overlay, 0.5, 0)[overlap]

    return panorama


def crop_black_border(image: ImageArray) -> ImageArray:
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return image

    x, y, width, height = cv2.boundingRect(coords)
    return image[y : y + height, x : x + width]


def draw_inlier_matches(
    left: ImageArray,
    right: ImageArray,
    left_keypoints: Sequence[cv2.KeyPoint],
    right_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    mask: ImageArray,
) -> ImageArray:
    matches_mask = mask.ravel().tolist() if mask is not None else None
    display_matches = [
        cv2.DMatch(
            _queryIdx=match.trainIdx,
            _trainIdx=match.queryIdx,
            _distance=match.distance,
        )
        for match in matches
    ]
    return cv2.drawMatches(
        left,
        left_keypoints,
        right,
        right_keypoints,
        display_matches,
        None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_mapped_points_on_left(
    left: ImageArray,
    left_keypoints: Sequence[cv2.KeyPoint],
    right_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    homography: ImageArray,
    mask: ImageArray,
) -> ImageArray:
    right_points = np.float32(
        [right_keypoints[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    left_points = np.float32(
        [left_keypoints[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    mapped_right_points = cv2.perspectiveTransform(right_points, homography)

    mapped_image = left.copy()
    inlier_mask = (
        np.ones(len(matches), dtype=bool)
        if mask is None
        else mask.ravel().astype(bool)
    )

    for left_point, mapped_point, is_inlier in zip(
        left_points.reshape(-1, 2),
        mapped_right_points.reshape(-1, 2),
        inlier_mask,
    ):
        if not is_inlier:
            continue

        left_x, left_y = np.round(left_point).astype(int)
        mapped_x, mapped_y = np.round(mapped_point).astype(int)

        cv2.circle(mapped_image, (left_x, left_y), 6, (0, 255, 0), 2)
        cv2.circle(mapped_image, (mapped_x, mapped_y), 4, (0, 0, 255), -1)
        cv2.line(mapped_image, (left_x, left_y), (mapped_x, mapped_y), (255, 0, 0), 1)

    return add_header(
        mapped_image,
        "H check: green=left, red=right->left, blue=error",
    )


def reconstruct_panorama(
    left: ImageArray,
    right: ImageArray,
    params: PanoramaReconstructionParams = PanoramaReconstructionParams(),
) -> PanoramaResult:
    """Map the right image into the left image coordinate system."""

    left = ensure_color_bgr(left)
    right = ensure_color_bgr(right)
    channel_name = normalize_channel_name(params.channel_name)
    params = PanoramaReconstructionParams(
        channel_name=channel_name,
        match_grid_rows=params.match_grid_rows,
        match_grid_cols=params.match_grid_cols,
        max_matches_per_grid_cell=params.max_matches_per_grid_cell,
        ratio_threshold=params.ratio_threshold,
        ransac_reprojection_threshold=params.ransac_reprojection_threshold,
    )

    left_keypoints, left_descriptors = detect_sift_features(left, channel_name)
    right_keypoints, right_descriptors = detect_sift_features(right, channel_name)
    matches = match_features(left_descriptors, right_descriptors, params.ratio_threshold)
    raw_match_count = len(matches)
    matches = balance_matches_by_grid(matches, right_keypoints, right.shape, params)
    homography, mask = estimate_right_to_left_homography(
        left_keypoints,
        right_keypoints,
        matches,
        params,
    )
    inlier_count = int(mask.sum()) if mask is not None else 0

    match_image = draw_inlier_matches(
        left,
        right,
        left_keypoints,
        right_keypoints,
        matches,
        mask,
    )
    mapped_points_image = draw_mapped_points_on_left(
        left,
        left_keypoints,
        right_keypoints,
        matches,
        homography,
        mask,
    )

    translation, panorama_size, left_offset = get_panorama_transform(
        left,
        right,
        homography,
    )
    warped_right = cv2.warpPerspective(right, translation @ homography, panorama_size)

    left_canvas = np.zeros_like(warped_right)
    offset_x, offset_y = left_offset
    height, width = left.shape[:2]
    left_canvas[offset_y : offset_y + height, offset_x : offset_x + width] = left

    panorama = blend_images(left_canvas, warped_right)
    panorama = crop_black_border(panorama)

    return PanoramaResult(
        panorama=panorama,
        warped_right=warped_right,
        match_visualization=match_image,
        mapped_points_visualization=mapped_points_image,
        raw_match_count=raw_match_count,
        balanced_match_count=len(matches),
        inlier_count=inlier_count,
        channel_name=channel_name,
    )
