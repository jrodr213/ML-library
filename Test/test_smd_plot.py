"""Tests for the SMD experiment comparison plot data loader."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Model.plot_smd_results import load_experiment


class SmdPlotTests(unittest.TestCase):
    def test_load_experiment_reads_metrics_configuration_and_history(self):
        """Saved result artifacts are converted into one plotting record."""
        with TemporaryDirectory() as temporary_directory:
            experiment_dir = Path(temporary_directory)
            (experiment_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "f1_score": 0.7,
                        "precision": 0.8,
                        "recall": 0.6,
                        "true_positives": 12,
                        "false_positives": 3,
                        "false_negatives": 4,
                        "mean_normal_window_reconstruction_error": 1.5,
                        "mean_anomalous_window_reconstruction_error": 8.5,
                    }
                )
            )
            (experiment_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 24,
                        "latent_size": 8,
                        "bottleneck_steps": 12,
                        "top_k_features": 3,
                        "top_k_timesteps": 8,
                    }
                )
            )
            (experiment_dir / "training_history.csv").write_text(
                "epoch,training_loss,validation_loss\n1,1.0,0.9\n"
            )

            experiment = load_experiment(experiment_dir)

            self.assertEqual(experiment["label"], "H24 L8 B12\nTop 3x8")
            self.assertEqual(experiment["history"][0]["validation_loss"], 0.9)


if __name__ == "__main__":
    unittest.main()
