from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    width: int = 224
    height: int = 224
    filter_type: str = "gaussian"
    kernel_size: int = 5
    sigma: float = 1.0


class Preprocessor:
    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()

    def _ensure_odd(self, size: int) -> int:
        return size if size % 2 == 1 else size + 1

    def process_path(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not read image: {path}")
        return self.process_array(image)

    def process_array(self, gray: np.ndarray) -> np.ndarray:
        if gray is None:
            raise ValueError("Input image is None")

        if gray.ndim != 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        cfg = self.config
        resized = cv2.resize(gray, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)

        kernel_size = self._ensure_odd(max(1, int(cfg.kernel_size)))
        if cfg.filter_type == "gaussian":
            filtered = cv2.GaussianBlur(resized, (kernel_size, kernel_size), cfg.sigma)
        elif cfg.filter_type == "median":
            filtered = cv2.medianBlur(resized, kernel_size)
        else:
            raise ValueError(f"Unsupported filter_type: {cfg.filter_type}")

        return filtered.astype(np.float32) / 255.0