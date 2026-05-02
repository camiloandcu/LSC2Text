import unittest
import tempfile
import os
import numpy as np
import cv2

from src.ml.preprocessing import Preprocessor, PreprocessConfig


def _write_temp_image(arr: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, arr)
    return path


class TestPreprocessing(unittest.TestCase):
    def test_deterministic_processing(self):
        np.random.seed(0)
        img = (np.random.rand(100, 100) * 255).astype(np.uint8)
        path = _write_temp_image(img)
        try:
            cfg = PreprocessConfig(width=64, height=64, filter_type="gaussian", kernel_size=3, sigma=0.5)
            p = Preprocessor(cfg)
            out1 = p.process_path(path)
            out2 = p.process_path(path)
            self.assertTrue(np.array_equal(out1, out2))
        finally:
            os.remove(path)

    def test_grayscale_resize_normalize(self):
        # Create a color image to ensure conversion path handles it
        color = np.zeros((50, 80, 3), dtype=np.uint8)
        color[..., 0] = 120
        path = _write_temp_image(color)
        try:
            cfg = PreprocessConfig(width=32, height=16, filter_type="gaussian", kernel_size=3)
            p = Preprocessor(cfg)
            out = p.process_path(path)
            self.assertEqual(out.shape, (16, 32))
            self.assertEqual(out.dtype, np.float32)
            self.assertGreaterEqual(out.min(), 0.0)
            self.assertLessEqual(out.max(), 1.0)
        finally:
            os.remove(path)

    def test_median_vs_gaussian_different(self):
        # Create a noisy image where median should differ
        img = np.zeros((60, 60), dtype=np.uint8)
        img[::2, ::2] = 255
        path = _write_temp_image(img)
        try:
            cfg_g = PreprocessConfig(width=32, height=32, filter_type="gaussian", kernel_size=5, sigma=1.0)
            cfg_m = PreprocessConfig(width=32, height=32, filter_type="median", kernel_size=5)
            pg = Preprocessor(cfg_g)
            pm = Preprocessor(cfg_m)
            out_g = pg.process_path(path)
            out_m = pm.process_path(path)
            self.assertFalse(np.array_equal(out_g, out_m))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
