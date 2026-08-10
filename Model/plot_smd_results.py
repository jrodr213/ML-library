"""Create a Matplotlib comparison of completed SMD experiment outputs."""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "ml-library-matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np


def build_parser():
    """Create command-line options for plotting saved SMD experiments."""
    parser = argparse.ArgumentParser(
        description="Compare completed SMD experiment metrics and loss histories."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        action="append",
        help="Completed experiment directory. May be supplied more than once.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Directory searched for experiment metrics when no directories are given.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/smd_experiment_comparison.png"),
        help="PNG path for the generated comparison figure.",
    )
    return parser


def discover_experiment_dirs(outputs_root):
    """Return completed experiment folders containing SMD metrics."""
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"outputs directory does not exist: {outputs_root}")
    experiment_dirs = sorted(
        metrics_path.parent
        for metrics_path in outputs_root.glob("*/metrics.json")
        if metrics_path.parent.name.startswith("smd_")
    )
    if not experiment_dirs:
        raise FileNotFoundError("no completed SMD experiment metrics were found")
    return experiment_dirs


def read_json(path):
    """Read a JSON artifact and raise a useful error if it is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"required experiment artifact is missing: {path}")
    with path.open() as file:
        return json.load(file)


def read_history(path):
    """Read the epoch, training-loss, and validation-loss CSV columns."""
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return [
            {
                "epoch": int(row["epoch"]),
                "training_loss": float(row["training_loss"]),
                "validation_loss": float(row["validation_loss"]),
            }
            for row in csv.DictReader(file)
        ]


def experiment_label(experiment_dir, config):
    """Build a compact label from saved model and scoring settings."""
    required_keys = ("hidden_size", "latent_size", "bottleneck_steps")
    if not all(key in config for key in required_keys):
        return experiment_dir.name

    label = (
        f"H{config['hidden_size']} L{config['latent_size']} "
        f"B{config['bottleneck_steps']}"
    )
    if "top_k_features" in config and "top_k_timesteps" in config:
        label += f"\nTop {config['top_k_features']}x{config['top_k_timesteps']}"
    if config.get("normalize_feature_errors"):
        label += "\nNormalized"
    return label


def load_experiment(experiment_dir):
    """Load metrics, configuration, and optional loss history for one run."""
    metrics = read_json(experiment_dir / "metrics.json")
    config = read_json(experiment_dir / "experiment_config.json")
    required_metrics = (
        "f1_score",
        "precision",
        "recall",
        "true_positives",
        "false_positives",
        "false_negatives",
        "mean_normal_window_reconstruction_error",
        "mean_anomalous_window_reconstruction_error",
    )
    missing_metrics = [name for name in required_metrics if name not in metrics]
    if missing_metrics:
        raise ValueError(
            f"metrics file is missing required values: {', '.join(missing_metrics)}"
        )
    return {
        "directory": experiment_dir,
        "label": experiment_label(experiment_dir, config),
        "metrics": metrics,
        "history": read_history(experiment_dir / "training_history.csv"),
    }


def add_value_labels(axis, bars, values, value_format):
    """Add concise value labels above a series of Matplotlib bars."""
    maximum = max(values) if values else 0
    offset = maximum * 0.02 if maximum else 0.02
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_experiment_results(experiments, output_path):
    """Save performance, error-count, loss, and score-separation comparisons."""
    labels = [experiment["label"] for experiment in experiments]
    metrics = [experiment["metrics"] for experiment in experiments]
    positions = np.arange(len(experiments))
    colors = plt.get_cmap("tab10")(np.arange(len(experiments)))

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle("SMD Machine-1-1 Experiment Comparison", fontsize=16)

    metric_axis = axes[0, 0]
    metric_names = ("F1", "Precision", "Recall")
    metric_keys = ("f1_score", "precision", "recall")
    width = 0.22
    for index, (name, key) in enumerate(zip(metric_names, metric_keys)):
        values = [float(metric[key]) for metric in metrics]
        bars = metric_axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=name,
        )
        add_value_labels(metric_axis, bars, values, "{:.3f}")
    metric_axis.set_title("Detection Metrics")
    metric_axis.set_ylabel("Score")
    metric_axis.set_ylim(0, 1.1)
    metric_axis.set_xticks(positions, labels)
    metric_axis.legend()
    metric_axis.grid(axis="y", alpha=0.25)

    count_axis = axes[0, 1]
    count_names = ("True positives", "False positives", "False negatives")
    count_keys = ("true_positives", "false_positives", "false_negatives")
    for index, (name, key) in enumerate(zip(count_names, count_keys)):
        values = [int(metric[key]) for metric in metrics]
        bars = count_axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=name,
        )
        add_value_labels(count_axis, bars, values, "{:d}")
    count_axis.set_title("Window Classification Counts")
    count_axis.set_ylabel("Windows")
    count_axis.set_xticks(positions, labels)
    count_axis.legend()
    count_axis.grid(axis="y", alpha=0.25)

    loss_axis = axes[1, 0]
    for color, experiment in zip(colors, experiments):
        history = experiment["history"]
        if history:
            loss_axis.plot(
                [record["epoch"] for record in history],
                [record["validation_loss"] for record in history],
                color=color,
                linewidth=2,
                label=experiment["label"].replace("\n", " "),
            )
    loss_axis.set_title("Validation Reconstruction Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("MSE")
    loss_axis.grid(alpha=0.25)
    if loss_axis.lines:
        loss_axis.legend(fontsize=8)
    else:
        loss_axis.text(0.5, 0.5, "No training histories found", ha="center", va="center")

    score_axis = axes[1, 1]
    normal_scores = [
        float(metric["mean_normal_window_reconstruction_error"])
        for metric in metrics
    ]
    anomalous_scores = [
        float(metric["mean_anomalous_window_reconstruction_error"])
        for metric in metrics
    ]
    normal_bars = score_axis.bar(
        positions - width / 2,
        normal_scores,
        width,
        label="Normal windows",
    )
    anomaly_bars = score_axis.bar(
        positions + width / 2,
        anomalous_scores,
        width,
        label="Anomalous windows",
    )
    add_value_labels(score_axis, normal_bars, normal_scores, "{:.1f}")
    add_value_labels(score_axis, anomaly_bars, anomalous_scores, "{:.1f}")
    score_axis.set_title("Mean Window Anomaly Score")
    score_axis.set_ylabel("Score (log scale)")
    score_axis.set_yscale("log")
    score_axis.set_xticks(positions, labels)
    score_axis.legend()
    score_axis.grid(axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    """Load saved results and write one PNG comparison figure."""
    args = build_parser().parse_args()
    experiment_dirs = args.experiment_dir or discover_experiment_dirs(args.outputs_root)
    experiments = [load_experiment(path) for path in experiment_dirs]
    plot_experiment_results(experiments, args.output)
    print(f"Saved SMD comparison plot to {args.output}")


if __name__ == "__main__":
    main()
