from pathlib import Path

import numpy as np

from panorama_reconstruction import api
from panorama_reconstruction.domain import ControlPointReconstructionParams, PanoramaResult


def test_sample_pair_is_packaged() -> None:
    pair = api.get_sample_image_paths()

    assert pair.left.exists()
    assert pair.right.exists()


def test_panorama_reconstruction_from_sample_pair(tmp_path: Path) -> None:
    pair = api.get_sample_image_paths()

    result = api.reconstruct_panorama_from_paths(pair.left, pair.right)

    assert isinstance(result, PanoramaResult)
    assert result.panorama.ndim == 3
    assert result.panorama.shape[2] == 3
    assert result.panorama.dtype == np.uint8
    assert result.raw_match_count >= result.balanced_match_count >= 4
    assert result.inlier_count >= 4

    outputs = api.save_reconstruction_outputs(result, tmp_path)
    assert outputs["panorama"].exists()
    assert outputs["feature_matches"].exists()


def textured_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = [40, 180, 230]
    image[36:44, :] = [255, 255, 255]
    return image


def identity_pairs():
    return [
        ((10.0, 10.0), (10.0, 10.0)),
        ((88.0, 12.0), (88.0, 12.0)),
        ((86.0, 68.0), (86.0, 68.0)),
        ((12.0, 66.0), (12.0, 66.0)),
    ]


def test_manual_reconstruction_from_control_points() -> None:
    image = textured_image()

    result = api.reconstruct_manual_panorama(image, image.copy(), identity_pairs())

    assert result.panorama.ndim == 3
    assert result.warped_right.ndim == 3
    assert result.method == "manual-perspective"
    assert result.inlier_count == 4


def test_manual_assisted_reconstruction_from_control_points() -> None:
    image = textured_image()
    params = ControlPointReconstructionParams(
        auto_grid_step=24,
        auto_template_radius=3,
        auto_search_radius=4,
        auto_min_score=0.9,
        auto_min_std=1.0,
        auto_max_points=8,
    )

    result = api.reconstruct_manual_assisted_panorama(
        image,
        image.copy(),
        identity_pairs(),
        params,
    )

    assert result.panorama.ndim == 3
    assert result.method == "manual-assisted-tps"
    assert result.extra_metrics["manual_pairs"] == 4
    assert result.extra_metrics["assisted_pairs"] > 0


def test_point_pair_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "points.json"

    api.save_point_pairs(path, identity_pairs())
    loaded = api.load_point_pairs(path)

    assert loaded == identity_pairs()
