from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from skimage.feature import hog, local_binary_pattern


@dataclass
class HogConfig:
    orientations: int = 10
    pixels_per_cell: Tuple[int, int] = (8, 8)
    cells_per_block: Tuple[int, int] = (1, 1)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = False


@dataclass
class LbpConfig:
    radius: int = 2
    n_points: int = 16
    method: str = "uniform"


@dataclass
class FeatureConfig:
    hog: HogConfig = field(default_factory=HogConfig)
    lbp: LbpConfig = field(default_factory=LbpConfig)
    version: str = "v1"


def _validate_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array")
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array")
    if image.size == 0:
        raise ValueError("Input image is empty")
    return image


def _lbp_bins(n_points: int, method: str) -> int:
    if method == "uniform":
        return n_points + 2
    if method == "nri_uniform":
        return n_points * (n_points - 1) + 2
    if method in {"default", "ror"}:
        return 2 ** n_points
    raise ValueError(f"Unsupported LBP method: {method}")


def extract_hog(image: np.ndarray, config: HogConfig) -> np.ndarray:
    image = _validate_image(image).astype(np.float32, copy=False)
    features = hog(
        image,
        orientations=config.orientations,
        pixels_per_cell=config.pixels_per_cell,
        cells_per_block=config.cells_per_block,
        block_norm=config.block_norm,
        transform_sqrt=config.transform_sqrt,
        feature_vector=True,
    )
    return np.asarray(features, dtype=np.float32)


def extract_lbp(image: np.ndarray, config: LbpConfig) -> np.ndarray:
    image = _validate_image(image).astype(np.float32, copy=False)
    lbp = local_binary_pattern(image, config.n_points, config.radius, method=config.method)
    bins = _lbp_bins(config.n_points, config.method)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    return hist.astype(np.float32)


def extract_features(image: np.ndarray, config: FeatureConfig | None = None) -> np.ndarray:
    cfg = config or FeatureConfig()
    hog_features = extract_hog(image, cfg.hog)
    lbp_features = extract_lbp(image, cfg.lbp)
    return np.concatenate([hog_features, lbp_features])
