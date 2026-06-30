"""Public facade for panorama reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..application import PanoramaReconstructionService, build_default_service
from ..configuration import PanoramaReconstructionConfig
from ..domain import (
    ControlPointReconstructionParams,
    ImageArray,
    ImagePairPaths,
    PanoramaReconstructionParams,
    PanoramaResult,
    PathLike,
    PointPair,
)


_default_service = None  # type: Optional[PanoramaReconstructionService]


def create_panorama_reconstruction_service(
    config: Optional[PanoramaReconstructionConfig] = None,
) -> PanoramaReconstructionService:
    return build_default_service(config or PanoramaReconstructionConfig())


def get_default_service() -> PanoramaReconstructionService:
    global _default_service
    if _default_service is None:
        _default_service = create_panorama_reconstruction_service()
    return _default_service


def load_image(path: PathLike) -> ImageArray:
    return get_default_service().load_image(path)


def save_image(image: ImageArray, path: PathLike) -> None:
    get_default_service().save_image(image, path)


def reconstruct_panorama(
    left: ImageArray,
    right: ImageArray,
    channel_name: str = "gray",
) -> PanoramaResult:
    return get_default_service().reconstruct(
        left,
        right,
        PanoramaReconstructionParams(channel_name=channel_name),
    )


def reconstruct_panorama_from_paths(
    left_path: PathLike,
    right_path: PathLike,
    channel_name: str = "gray",
) -> PanoramaResult:
    return get_default_service().reconstruct_from_paths(
        left_path,
        right_path,
        PanoramaReconstructionParams(channel_name=channel_name),
    )


def reconstruct_manual_panorama(
    left: ImageArray,
    right: ImageArray,
    point_pairs: List[PointPair],
) -> PanoramaResult:
    return get_default_service().reconstruct_from_points(left, right, point_pairs)


def reconstruct_manual_assisted_panorama(
    left: ImageArray,
    right: ImageArray,
    point_pairs: List[PointPair],
    params: Optional[ControlPointReconstructionParams] = None,
) -> PanoramaResult:
    return get_default_service().reconstruct_assisted_from_points(
        left,
        right,
        point_pairs,
        params,
    )


def save_reconstruction_outputs(
    result: PanoramaResult,
    output_dir: PathLike,
) -> Dict[str, Path]:
    return get_default_service().save_outputs(result, output_dir)


def load_point_pairs(path: PathLike) -> List[PointPair]:
    return get_default_service().load_point_pairs(path)


def save_point_pairs(path: PathLike, point_pairs: List[PointPair]) -> None:
    get_default_service().save_point_pairs(path, point_pairs)


def get_sample_image_paths() -> ImagePairPaths:
    return get_default_service().get_sample_image_paths()
