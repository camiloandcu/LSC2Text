import unittest

import numpy as np

from src.ml.features.feature_extraction import FeatureConfig, LbpConfig, extract_features


class TestFeatureExtraction(unittest.TestCase):
    def test_vector_length_consistent(self):
        image_a = (np.random.rand(64, 64) * 255).astype(np.float32)
        image_b = (np.random.rand(64, 64) * 255).astype(np.float32)
        config = FeatureConfig()

        vec_a = extract_features(image_a, config)
        vec_b = extract_features(image_b, config)

        self.assertEqual(vec_a.shape, vec_b.shape)
        self.assertEqual(len(vec_a.shape), 1)

    def test_invalid_image_shape(self):
        config = FeatureConfig(lbp=LbpConfig())

        with self.assertRaises(ValueError):
            extract_features(None, config)

        with self.assertRaises(ValueError):
            extract_features(np.zeros((10, 10, 3), dtype=np.float32), config)

        with self.assertRaises(ValueError):
            extract_features(np.array([], dtype=np.float32), config)


if __name__ == "__main__":
    unittest.main()
