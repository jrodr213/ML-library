"""SMD window metadata and reconstruction-score tests."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from Model.run_smd_experiment import (
    create_test_window_records,
    create_window_view,
    normalize_feature_errors,
    reconstruction_score,
    rolling_mad_predictions,
    write_training_history,
    write_window_scores,
)


class SmdExperimentTests(unittest.TestCase):
    def test_window_shape_and_point_label_aggregation(self):
        values = np.arange(80, dtype=np.float64).reshape(20, 4)
        windows = create_window_view(values, window_size=8, step_size=3)
        labels = np.zeros(20, dtype=np.int64)
        labels[4] = 1
        records = create_test_window_records(labels, window_size=8, step_size=3)

        self.assertEqual(windows.shape, (5, 8, 4))
        self.assertEqual(records[0]["actual_anomaly"], 1)
        self.assertEqual(records[1]["anomalous_timesteps"], 1)
        self.assertEqual(records[-1]["actual_anomaly"], 0)

    def test_top_k_score_uses_worst_sensors_then_timesteps(self):
        """A sparse sensor error remains visible in the final window score."""
        clean = np.zeros((4, 2), dtype=np.float64)
        reconstruction = np.array([[0, 0], [1, 10], [3, 3], [5, 5]], dtype=np.float64)
        self.assertEqual(
            reconstruction_score(reconstruction, clean, "mean", 2, 1),
            4.5,
        )
        self.assertEqual(
            reconstruction_score(reconstruction, clean, "top_k", 2, 1),
            7.5,
        )

    def test_feature_error_normalization_uses_validation_median_and_mad(self):
        """Sensor errors become comparable before the top-sensor score is calculated."""
        normalized = normalize_feature_errors(
            np.array([[3.0, 14.0], [0.0, 9.0]]),
            {
                "median": np.array([1.0, 10.0]),
                "robust_scale": np.array([2.0, 2.0]),
            },
        )
        np.testing.assert_allclose(normalized, [[1.0, 2.0], [0.0, 0.0]])

    def test_output_writers_create_csv_artifacts(self):
        """CSV artifact writers accept paths and preserve their column headers."""
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            history_path = output_dir / "training_history.csv"
            scores_path = output_dir / "window_scores.csv"
            records = create_test_window_records(
                np.array([0, 1, 0, 0], dtype=np.int64),
                window_size=2,
                step_size=1,
            )

            write_training_history(
                history_path,
                [{"epoch": 1, "training_loss": 0.5, "validation_loss": 0.4}],
            )
            write_window_scores(scores_path, records, [0.1, 0.2, 0.3], 0.25)

            self.assertIn("validation_loss", history_path.read_text())
            self.assertIn("reconstruction_score", scores_path.read_text())

    def test_rolling_mad_uses_warmup_and_rejects_anomalous_scores(self):
        """Rolling calibration excludes warm-up and flagged scores from its baseline."""
        thresholds, predictions, calibration_windows = rolling_mad_predictions(
            [1.0, 1.1, 0.9, 1.0, 8.0, 1.1],
            warmup_windows=3,
            history_windows=3,
            mad_multiplier=3.0,
        )

        self.assertTrue(np.all(calibration_windows[:3]))
        self.assertTrue(np.all(predictions[:3] == -1))
        self.assertTrue(np.all(np.isnan(thresholds[:3])))
        self.assertEqual(predictions[4], 1)
        self.assertLess(thresholds[5], 2.0)


if __name__ == "__main__":
    unittest.main()
