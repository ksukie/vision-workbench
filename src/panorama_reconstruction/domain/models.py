"""Domain models shared by the panorama reconstruction layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np


ImageArray = np.ndarray
PathLike = Union[str, Path]
Point = Tuple[float, float]
PointPair = Tuple[Point, Point]


@dataclass(frozen=True)
class ImagePairPaths:
    """A left/right image pair used by the panorama workflow."""

    left: Path
    right: Path


@dataclass(frozen=True)
class PanoramaReconstructionParams:
    """User-adjustable parameters for panorama reconstruction."""

    channel_name: str = "gray"
    match_grid_rows: int = 4
    match_grid_cols: int = 4
    max_matches_per_grid_cell: int = 12
    ratio_threshold: float = 0.75
    ransac_reprojection_threshold: float = 5.0


@dataclass(frozen=True)
class ControlPointReconstructionParams:
    """Parameters for manual and assisted control-point reconstruction."""

    auto_grid_step: int = 32
    auto_template_radius: int = 9
    auto_search_radius: int = 28
    auto_min_score: float = 0.72
    auto_min_std: float = 6.0
    auto_max_points: int = 220
    tps_smooth: float = 1e-4
    tps_padding: int = 16
    tps_boundary_step: int = 24


@dataclass(frozen=True)
class PanoramaResult:
    """Images and metrics returned by the panorama algorithm."""

    panorama: ImageArray
    warped_right: ImageArray
    match_visualization: ImageArray
    mapped_points_visualization: ImageArray
    raw_match_count: int
    balanced_match_count: int
    inlier_count: int
    channel_name: str
    method: str = "automatic"
    extra_metrics: Dict[str, object] = field(default_factory=dict)

    def metrics(self) -> Dict[str, object]:
        metrics = {
            "method": self.method,
            "channel": self.channel_name,
            "raw_matches": self.raw_match_count,
            "balanced_matches": self.balanced_match_count,
            "inliers": self.inlier_count,
            "panorama_shape": tuple(int(value) for value in self.panorama.shape),
        }
        metrics.update(self.extra_metrics)
        return metrics
