import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna

from src.ml.optimization import (
    OptimizationConfig,
    create_objective,
    params_from_trial,
    run_optimization,
    sample_svm_params,
)


class TestHyperparamOptimization(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.array(
            [
                [-2.0, -1.5],
                [-1.8, -1.0],
                [-1.2, -1.7],
                [1.0, 1.3],
                [1.4, 1.6],
                [1.8, 1.1],
            ],
            dtype=np.float32,
        )
        self.labels = np.array(["A", "A", "A", "B", "B", "B"])

    def test_optimization_loop_selects_best_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = OptimizationConfig(
                model_type="svm",
                n_trials=2,
                seed=123,
                output_dir=root / "experiments",
                registry_dir=root / "registry",
                pruner_startup_trials=1,
            )

            study, artifacts = run_optimization(
                self.features,
                self.labels,
                config,
                validation_features=self.features,
                validation_labels=self.labels,
                storage_url="sqlite:///:memory:",
            )

            self.assertEqual(len(study.trials), 2)
            self.assertIsNotNone(study.best_trial)
            self.assertTrue(artifacts["best_params"].exists())
            self.assertTrue(artifacts["trial_log"].exists())
            self.assertTrue(artifacts["model"].exists())

            payload = json.loads(artifacts["best_params"].read_text(encoding="utf-8"))
            self.assertEqual(payload["model_type"], "svm")
            self.assertIn("params", payload)
            self.assertNotIn("kernel", payload["params"])
            self.assertNotIn("kernel", payload["trial_params"])

    def test_invalid_param_combos_are_pruned(self):
        config = OptimizationConfig(model_type="svm", n_trials=1)

        def invalid_space(_trial):
            return {"C": -1.0}

        objective = create_objective(
            self.features,
            self.labels,
            self.features,
            self.labels,
            config,
            search_space=invalid_space,
        )

        with self.assertRaises(optuna.exceptions.TrialPruned):
            objective(optuna.trial.FixedTrial({}))

    def test_svm_search_space_omits_kernel(self):
        trial = optuna.trial.FixedTrial({"C": 1.0})
        params = sample_svm_params(trial)

        self.assertEqual(params, {"C": 1.0})
        self.assertNotIn("kernel", params_from_trial("svm", params))


if __name__ == "__main__":
    unittest.main()
