"""Application service for panorama reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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
from ..infrastructure import OpenCvImageRepository
from ..processing import (
    create_channel_preview,
    load_point_pairs,
    reconstruct_manual,
    reconstruct_manual_assisted,
    reconstruct_panorama,
    save_point_pairs,
)


class PanoramaReconstructionService:
    """Coordinates image persistence and the panorama reconstruction algorithm."""

    def __init__(
        self,
        repository: OpenCvImageRepository,
        config: PanoramaReconstructionConfig = PanoramaReconstructionConfig(),
    ) -> None:
        self._repository = repository
        self._config = config

    def load_image(self, path: PathLike) -> ImageArray:
        return self._repository.load(path)

    def save_image(self, image: ImageArray, path: PathLike) -> None:
        self._repository.save(image, path)

    def create_channel_preview(self, left: ImageArray, right: ImageArray) -> ImageArray:
        return create_channel_preview(left, right)

    def reconstruct(
        self,
        left: ImageArray,
        right: ImageArray,
        params: Optional[PanoramaReconstructionParams] = None,
    ) -> PanoramaResult:
        return reconstruct_panorama(left, right, params or PanoramaReconstructionParams())

    def reconstruct_from_paths(
        self,
        left_path: PathLike,
        right_path: PathLike,
        params: Optional[PanoramaReconstructionParams] = None,
    ) -> PanoramaResult:
        left = self.load_image(left_path)
        right = self.load_image(right_path)
        return self.reconstruct(left, right, params)

    def reconstruct_from_points(
        self,
        left: ImageArray,
        right: ImageArray,
        point_pairs: List[PointPair],
    ) -> PanoramaResult:
        return reconstruct_manual(left, right, point_pairs)

    def reconstruct_assisted_from_points(
        self,
        left: ImageArray,
        right: ImageArray,
        point_pairs: List[PointPair],
        params: Optional[ControlPointReconstructionParams] = None,
    ) -> PanoramaResult:
        return reconstruct_manual_assisted(
            left,
            right,
            point_pairs,
            params or ControlPointReconstructionParams(),
        )

    def load_point_pairs(self, path: PathLike) -> List[PointPair]:
        return load_point_pairs(path)

    def save_point_pairs(self, path: PathLike, point_pairs: List[PointPair]) -> None:
        save_point_pairs(path, point_pairs)

    def save_outputs(self, result: PanoramaResult, output_dir: PathLike) -> Dict[str, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        outputs = {
            "panorama": directory / "panorama.png",
            "right_warped_to_left": directory / "right_warped_to_left.png",
            "feature_matches": directory / "feature_matches.png",
            "mapped_points_on_left": directory / "mapped_points_on_left.png",
        }
        self.save_image(result.panorama, outputs["panorama"])
        self.save_image(result.warped_right, outputs["right_warped_to_left"])
        self.save_image(result.match_visualization, outputs["feature_matches"])
        self.save_image(result.mapped_points_visualization, outputs["mapped_points_on_left"])
        return outputs

    def get_sample_image_paths(self) -> ImagePairPaths:
        asset_dir = Path(__file__).resolve().parents[1] / "assets" / "buildingImgs"
        return ImagePairPaths(left=asset_dir / "left.png", right=asset_dir / "right.png")


def build_default_service(
    config: PanoramaReconstructionConfig = PanoramaReconstructionConfig(),
) -> PanoramaReconstructionService:
    return PanoramaReconstructionService(repository=OpenCvImageRepository(), config=config)
