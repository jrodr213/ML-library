"""Train and evaluate the handwritten temporal LSTM autoencoder on SMD."""

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "ml-library-matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from Model.Autoencoder import Autoencoder
from Model.Run import load_model, save_model


DEFAULT_SEED = 42
DEFAULT_GRADIENT_CLIP_VALUE = 1.0


def build_parser():
    """Create command-line options for one SMD machine experiment."""
    parser = argparse.ArgumentParser(
        description="Train the custom temporal autoencoder on one SMD machine."
    )
    parser.add_argument(
        "--smd-root",
        type=Path,
        default=Path("data/OmniAnomaly/ServerMachineDataset"),
    )
    parser.add_argument("--machine", default="machine-1-1")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--latent-size", type=int, default=4)
    parser.add_argument("--bottleneck-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--train-split", type=float, default=0.80)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--threshold-mode",
        choices=["static", "rolling_mad"],
        default="rolling_mad",
        help="Use a fixed validation percentile or an adaptive rolling-MAD threshold.",
    )
    parser.add_argument(
        "--warmup-windows",
        type=int,
        default=256,
        help="Initial assumed-normal windows used only to calibrate rolling-MAD scores.",
    )
    parser.add_argument(
        "--rolling-history-windows",
        type=int,
        default=512,
        help="Maximum trusted-normal scores retained by rolling-MAD calibration.",
    )
    parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=6.0,
        help="Number of robust score deviations allowed before an anomaly alert.",
    )
    parser.add_argument("--feature-count", type=int, default=38)
    parser.add_argument(
        "--score-method",
        choices=["mean", "top_k"],
        default="top_k",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=3,
        help="Number of highest-error sensors averaged at each timestep.",
    )
    parser.add_argument(
        "--top-k-timesteps",
        type=int,
        default=8,
        help="Number of highest-error timestep scores averaged per window.",
    )
    parser.add_argument(
        "--normalize-feature-errors",
        action="store_true",
        help="Normalize each sensor error using median and MAD from validation data.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/smd_machine_1_1"),
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="Resume only from a compatible SMD checkpoint created by this runner.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run the required 5-feature and 38-feature model smoke tests only.",
    )
    return parser


def configure_logging():
    """Configure consistent console logging for SMD experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def validate_arguments(args):
    """Validate SMD dimensions, scoring, and training settings."""
    positive_names = [
        "window_size",
        "step_size",
        "hidden_size",
        "latent_size",
        "bottleneck_steps",
        "epochs",
        "feature_count",
        "top_k_features",
        "top_k_timesteps",
        "warmup_windows",
        "rolling_history_windows",
    ]
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be greater than zero")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("learning-rate must be a finite number greater than zero")
    if not 0 < args.train_split < 1:
        raise ValueError("train-split must be between zero and one")
    if not 0 < args.threshold_percentile < 100:
        raise ValueError("threshold-percentile must be between zero and 100")
    if not math.isfinite(args.mad_multiplier) or args.mad_multiplier <= 0:
        raise ValueError("mad-multiplier must be a finite number greater than zero")
    if args.bottleneck_steps > args.window_size:
        raise ValueError("bottleneck-steps cannot exceed window-size")
    if args.top_k_timesteps > args.window_size:
        raise ValueError("top-k-timesteps cannot exceed window-size")
    if args.top_k_features > args.feature_count:
        raise ValueError("top-k-features cannot exceed feature-count")
    if not args.machine or Path(args.machine).name != args.machine:
        raise ValueError("machine must be a simple SMD machine filename stem")


def machine_paths(smd_root, machine):
    """Return the train, test, and point-label paths for one machine."""
    return {
        "train": smd_root / "train" / f"{machine}.txt",
        "test": smd_root / "test" / f"{machine}.txt",
        "labels": smd_root / "test_label" / f"{machine}.txt",
    }


def load_smd_machine(smd_root, machine):
    """Load one machine's multivariate data and binary point labels."""
    paths = machine_paths(smd_root, machine)
    missing_paths = [path for path in paths.values() if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"SMD machine files are missing: {missing}")

    training = np.genfromtxt(paths["train"], delimiter=",", dtype=np.float64)
    testing = np.genfromtxt(paths["test"], delimiter=",", dtype=np.float64)
    labels = np.genfromtxt(paths["labels"], delimiter=",", dtype=np.float64)

    if training.ndim != 2 or testing.ndim != 2:
        raise ValueError("SMD training and test data must both be two-dimensional")
    if training.shape[1] != testing.shape[1]:
        raise ValueError("SMD training and test data must have the same feature count")
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    if labels.ndim != 1 or len(labels) != len(testing):
        raise ValueError("SMD point labels must have one value for every test timestep")
    if not np.isfinite(training).all() or not np.isfinite(testing).all():
        raise ValueError("SMD feature data contains non-finite values")
    if not np.isfinite(labels).all():
        raise ValueError("SMD labels contain non-finite values")

    return training, testing, (labels != 0).astype(np.int64), paths


def select_features(training, testing, feature_count):
    """Select the same leading feature columns from train and test data."""
    available_features = training.shape[1]
    if feature_count > available_features:
        raise ValueError(
            f"feature-count ({feature_count}) exceeds available SMD features "
            f"({available_features})"
        )
    return training[:, :feature_count].copy(), testing[:, :feature_count].copy()


def split_training_timeline(training, train_split):
    """Chronologically divide raw training data into train and validation parts."""
    # Split raw timesteps first so train and validation windows cannot overlap.
    split_index = int(len(training) * train_split)
    if split_index == 0 or split_index == len(training):
        raise ValueError("train-split must leave both training and validation timesteps")
    return training[:split_index], training[split_index:]


def fit_scaler(training):
    """Fit finite per-feature scaling statistics using training data only."""
    mean = np.mean(training, axis=0)
    scale = np.std(training, axis=0)
    # Constant features are centered and left with a unit divisor.
    scale = np.where(scale < 1e-12, 1.0, scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("fitted scaler parameters must be finite")
    return mean, scale


def transform(values, mean, scale):
    """Standardize feature values and reject non-finite results."""
    transformed = (values - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("scaled SMD values must be finite")
    return transformed


def create_window_view(values, window_size, step_size):
    """Create a memory-efficient view of equally sized sliding windows."""
    if values.ndim != 2:
        raise ValueError("window source values must be two-dimensional")
    if len(values) < window_size:
        raise ValueError("timeline does not contain one complete window")
    # This is a view rather than a copied 3D array, which keeps SMD memory use
    # manageable while preserving the [window, timestep, feature] shape.
    all_windows = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=window_size,
        axis=0,
    )
    windows = np.moveaxis(all_windows, -1, 1)[::step_size]
    if windows.ndim != 3 or windows.shape[1:] != (window_size, values.shape[1]):
        raise ValueError("generated SMD windows have an unexpected shape")
    return windows


def create_test_window_records(test_labels, window_size, step_size):
    """Aggregate point labels into metadata for each test window."""
    if len(test_labels) < window_size:
        raise ValueError("test labels do not contain one complete window")
    starts = np.arange(0, len(test_labels) - window_size + 1, step_size, dtype=int)
    # A cumulative sum gives each window's label count without repeatedly
    # scanning all of its point labels.
    cumulative_labels = np.concatenate(([0], np.cumsum(test_labels, dtype=int)))
    anomalous_counts = cumulative_labels[starts + window_size] - cumulative_labels[starts]
    return [
        {
            "window_index": int(index),
            "start_timestep": int(start),
            "end_timestep": int(start + window_size - 1),
            "actual_anomaly": int(count > 0),
            "anomalous_timesteps": int(count),
            "anomalous_fraction": float(count / window_size),
        }
        for index, (start, count) in enumerate(zip(starts, anomalous_counts))
    ]


def reconstruction_score(
    reconstruction,
    clean_window,
    score_method,
    top_k_timesteps,
    top_k_features,
    feature_error_normalizer=None,
):
    """Score a window from its highest-error sensors and timesteps."""
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    clean_window = np.asarray(clean_window, dtype=np.float64)
    if reconstruction.shape != clean_window.shape or reconstruction.ndim != 2:
        raise ValueError("reconstruction must match a two-dimensional clean window")
    if top_k_features <= 0:
        raise ValueError("top_k_features must be greater than zero")
    if top_k_timesteps <= 0:
        raise ValueError("top_k_timesteps must be greater than zero")

    feature_errors = normalize_feature_errors(
        np.abs(reconstruction - clean_window),
        feature_error_normalizer,
    )
    feature_k = min(top_k_features, feature_errors.shape[1])
    timestep_error = np.mean(
        np.partition(feature_errors, -feature_k, axis=1)[:, -feature_k:],
        axis=1,
    )
    if score_method == "mean":
        return float(np.mean(timestep_error))
    k = min(top_k_timesteps, len(timestep_error))
    return float(np.mean(np.partition(timestep_error, -k)[-k:]))


def score_windows(
    model,
    windows,
    score_method,
    top_k_timesteps,
    top_k_features,
    feature_error_normalizer=None,
):
    """Reconstruct every window and return one anomaly score per window."""
    scores = []
    for window in windows:
        clean_window = window.tolist()
        reconstruction = model.reconstruct(clean_window)
        scores.append(
            reconstruction_score(
                reconstruction,
                clean_window,
                score_method,
                top_k_timesteps,
                top_k_features,
                feature_error_normalizer,
            )
        )
    return np.asarray(scores, dtype=np.float64)


def fit_feature_error_normalizer(model, validation_windows):
    """Fit per-sensor reconstruction-error median and MAD from normal validation data."""
    validation_errors = []
    for window in validation_windows:
        reconstruction = np.asarray(model.reconstruct(window.tolist()), dtype=np.float64)
        if reconstruction.shape != window.shape:
            raise ValueError("validation reconstruction must match its input window")
        validation_errors.append(np.abs(reconstruction - window))

    if not validation_errors:
        raise ValueError("validation windows are required for feature-error normalization")
    errors = np.concatenate(validation_errors, axis=0)
    median = np.median(errors, axis=0)
    mad = np.median(np.abs(errors - median), axis=0)
    robust_scale = np.maximum(1.4826 * mad, 1e-12)
    return {
        "median": median,
        "mad": mad,
        "robust_scale": robust_scale,
    }


def normalize_feature_errors(feature_errors, feature_error_normalizer=None):
    """Convert raw sensor errors into non-negative robust deviations when enabled."""
    feature_errors = np.asarray(feature_errors, dtype=np.float64)
    if feature_errors.ndim != 2:
        raise ValueError("feature errors must be a two-dimensional timestep-by-feature array")
    if feature_error_normalizer is None:
        return feature_errors

    median = np.asarray(feature_error_normalizer["median"], dtype=np.float64)
    robust_scale = np.asarray(
        feature_error_normalizer["robust_scale"],
        dtype=np.float64,
    )
    if median.shape != (feature_errors.shape[1],):
        raise ValueError("feature-error median must match the number of input features")
    if robust_scale.shape != median.shape:
        raise ValueError("feature-error robust scale must match the median shape")
    if not np.isfinite(median).all() or not np.isfinite(robust_scale).all():
        raise ValueError("feature-error normalizer values must be finite")
    if np.any(robust_scale <= 0):
        raise ValueError("feature-error robust scales must be greater than zero")
    return np.maximum((feature_errors - median) / robust_scale, 0.0)


def rolling_mad_predictions(
    scores,
    warmup_windows,
    history_windows,
    mad_multiplier,
):
    """Return adaptive thresholds and predictions from trusted normal scores.

    Warm-up scores seed the initial baseline and are excluded from predictions.
    Later anomalous scores are not added to the normal history, preventing them
    from raising future thresholds.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("rolling-MAD calibration requires one or more scores")
    if not np.isfinite(scores).all():
        raise ValueError("rolling-MAD calibration scores must be finite")
    if warmup_windows <= 0 or warmup_windows >= len(scores):
        raise ValueError(
            "warmup-windows must be greater than zero and smaller than test windows"
        )
    if history_windows <= 0:
        raise ValueError("rolling-history-windows must be greater than zero")
    if not math.isfinite(mad_multiplier) or mad_multiplier <= 0:
        raise ValueError("mad-multiplier must be a finite number greater than zero")

    thresholds = np.full(len(scores), np.nan, dtype=np.float64)
    predictions = np.full(len(scores), -1, dtype=np.int64)
    calibration_windows = np.zeros(len(scores), dtype=bool)
    calibration_windows[:warmup_windows] = True
    trusted_normal_scores = list(scores[:warmup_windows])

    for index in range(warmup_windows, len(scores)):
        history = np.asarray(trusted_normal_scores[-history_windows:], dtype=np.float64)
        center = float(np.median(history))
        mad = float(np.median(np.abs(history - center)))
        robust_spread = max(1.4826 * mad, 1e-12)
        threshold = center + mad_multiplier * robust_spread
        prediction = int(scores[index] > threshold)

        thresholds[index] = threshold
        predictions[index] = prediction
        if not prediction:
            trusted_normal_scores.append(float(scores[index]))

    return thresholds, predictions, calibration_windows


def build_model(args, input_size):
    """Create the temporal autoencoder for the selected SMD feature count."""
    return Autoencoder(
        input_size=input_size,
        learning_rate=args.learning_rate,
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        bottleneck_steps=args.bottleneck_steps,
        window_size=args.window_size,
        max_epochs=1,
        gradient_clip_value=DEFAULT_GRADIENT_CLIP_VALUE,
    )


def load_smd_checkpoint(checkpoint_path, args):
    """Load a matching SMD checkpoint and reject legacy or NAB checkpoints."""
    payload = load_model(
        checkpoint_path,
        expected_bottleneck_steps=args.bottleneck_steps,
    )
    metadata = payload.get("metadata", {})
    if metadata.get("dataset") != "smd":
        raise ValueError(
            "checkpoint is not an SMD checkpoint; NAB and other dataset checkpoints "
            "cannot be used for this experiment"
        )
    expected_values = {
        "machine": args.machine,
        "input_size": args.feature_count,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "window_size": args.window_size,
    }
    for name, expected_value in expected_values.items():
        actual_value = metadata.get(name)
        if actual_value != expected_value:
            raise ValueError(
                f"checkpoint {name} is incompatible: expected {expected_value!r}, "
                f"found {actual_value!r}"
            )
    return payload["model"]


def mean_reconstruction_loss(model, windows):
    """Return average MSE across a set of clean validation windows."""
    losses = []
    for window in windows:
        clean_window = window.tolist()
        reconstruction = np.asarray(model.reconstruct(clean_window), dtype=np.float64)
        losses.append(float(np.mean((reconstruction - window) ** 2)))
    return float(np.mean(losses))


def train_model(model, training_windows, validation_windows, epochs):
    """Train by epoch and retain the model with the best validation loss."""
    history = []
    best_model = None
    best_validation_loss = math.inf

    for epoch in range(1, epochs + 1):
        training_losses = []
        for window in training_windows:
            clean_window = window.tolist()
            reconstruction = model.run(clean_window, target=1)
            training_losses.append(
                float(np.mean((np.asarray(reconstruction) - window) ** 2))
            )

        validation_loss = mean_reconstruction_loss(model, validation_windows)
        epoch_record = {
            "epoch": epoch,
            "training_loss": float(np.mean(training_losses)),
            "validation_loss": validation_loss,
        }
        history.append(epoch_record)
        logging.info(
            "epoch %d/%d: training_loss=%.8f validation_loss=%.8f",
            epoch,
            epochs,
            epoch_record["training_loss"],
            validation_loss,
        )
        # Keep a separate snapshot so later epochs cannot overwrite the best
        # validation model before test scoring.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

    return best_model, history, best_validation_loss


def safe_binary_metrics(actual, predicted, scores):
    """Calculate classification metrics while handling a one-class label set."""
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    true_positives = int(np.sum((predicted == 1) & (actual == 1)))
    false_positives = int(np.sum((predicted == 1) & (actual == 0)))
    true_negatives = int(np.sum((predicted == 0) & (actual == 0)))
    false_negatives = int(np.sum((predicted == 0) & (actual == 1)))
    metrics = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
        "window_accuracy": float(accuracy_score(actual, predicted)),
    }
    if len(np.unique(actual)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(actual, scores))
        metrics["pr_auc"] = float(average_precision_score(actual, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def write_json(path, contents):
    """Write structured experiment output as formatted JSON."""
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(contents, output_file, indent=2, sort_keys=True)


def write_training_history(path, history):
    """Write per-epoch training and validation losses to CSV."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["epoch", "training_loss", "validation_loss"],
        )
        writer.writeheader()
        writer.writerows(history)


def write_window_scores(
    path,
    records,
    scores,
    thresholds,
    predictions=None,
    calibration_windows=None,
):
    """Write window metadata, scores, labels, and predictions to CSV."""
    scores = np.asarray(scores, dtype=np.float64)
    if np.isscalar(thresholds):
        thresholds = np.full(len(scores), float(thresholds), dtype=np.float64)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)
    if predictions is None:
        predictions = (scores >= thresholds).astype(np.int64)
    else:
        predictions = np.asarray(predictions, dtype=np.int64)
    if calibration_windows is None:
        calibration_windows = np.zeros(len(scores), dtype=bool)
    else:
        calibration_windows = np.asarray(calibration_windows, dtype=bool)
    if not (
        len(records)
        == len(scores)
        == len(thresholds)
        == len(predictions)
        == len(calibration_windows)
    ):
        raise ValueError("window score records, thresholds, and predictions must align")

    fields = [
        "window_index",
        "start_timestep",
        "end_timestep",
        "reconstruction_score",
        "threshold",
        "predicted_anomaly",
        "calibration_window",
        "actual_anomaly",
        "anomalous_timesteps",
        "anomalous_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for record, score, threshold, prediction, calibration_window in zip(
            records,
            scores,
            thresholds,
            predictions,
            calibration_windows,
        ):
            writer.writerow(
                {
                    **record,
                    "reconstruction_score": float(score),
                    "threshold": "" if np.isnan(threshold) else float(threshold),
                    "predicted_anomaly": "" if prediction < 0 else int(prediction),
                    "calibration_window": int(calibration_window),
                }
            )


def save_plots(
    output_dir,
    history,
    records,
    scores,
    thresholds,
    calibration_windows=None,
):
    """Save training-loss and test-score plots for one experiment."""
    scores = np.asarray(scores, dtype=np.float64)
    if np.isscalar(thresholds):
        thresholds = np.full(len(scores), float(thresholds), dtype=np.float64)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)
    if calibration_windows is None:
        calibration_windows = np.zeros(len(scores), dtype=bool)
    else:
        calibration_windows = np.asarray(calibration_windows, dtype=bool)
    epochs = [record["epoch"] for record in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [record["training_loss"] for record in history], label="training")
    plt.plot(epochs, [record["validation_loss"] for record in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared reconstruction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_plot.png", dpi=140)
    plt.close()

    window_indices = [record["window_index"] for record in records]
    actual = [record["actual_anomaly"] for record in records]
    plt.figure(figsize=(10, 4))
    plt.plot(window_indices, scores, linewidth=0.8, label="reconstruction score")
    valid_thresholds = ~np.isnan(thresholds)
    if np.any(valid_thresholds):
        plt.plot(
            np.asarray(window_indices)[valid_thresholds],
            thresholds[valid_thresholds],
            color="tab:red",
            linestyle="--",
            label="threshold",
        )
    if np.any(calibration_windows):
        calibration_indices = np.asarray(window_indices)[calibration_windows]
        plt.axvspan(
            calibration_indices[0],
            calibration_indices[-1],
            color="tab:gray",
            alpha=0.15,
            label="calibration",
        )
    anomaly_indices = [index for index, label in zip(window_indices, actual) if label]
    anomaly_scores = [score for score, label in zip(scores, actual) if label]
    if anomaly_indices:
        plt.scatter(anomaly_indices, anomaly_scores, s=8, color="tab:orange", label="actual anomaly")
    plt.xlabel("Test window index")
    plt.ylabel("Window reconstruction score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_error_plot.png", dpi=140)
    plt.close()


def run_smoke_test(feature_count):
    """Verify forward, backward, clipping, and updates for one feature width."""
    smoke_args = argparse.Namespace(
        learning_rate=0.003,
        latent_size=4,
        hidden_size=8,
        bottleneck_steps=16,
        window_size=128,
    )
    random_generator = np.random.default_rng(DEFAULT_SEED + feature_count)
    batch = random_generator.normal(size=(2, 128, feature_count))
    model = build_model(smoke_args, feature_count)
    reconstruction = np.asarray(model.reconstruct_batch(batch.tolist()), dtype=np.float64)
    expected_shapes = {
        "input": (128, feature_count),
        "encoder": (128, 8),
        "compressed": (16, 8),
        "bottleneck": (16, 4),
        "expanded": (128, 8),
        "reconstruction": (128, feature_count),
    }
    if reconstruction.shape != batch.shape or model.last_temporal_shapes != expected_shapes:
        raise AssertionError("temporal autoencoder smoke-test shapes are incorrect")

    compressor_weights = [row[:] for row in model.temporal_compressor.weights]
    decoded = model.run(batch[0].tolist(), target=1)
    decoded_array = np.asarray(decoded, dtype=np.float64)
    if decoded_array.shape != (128, feature_count):
        raise AssertionError("smoke-test reconstruction has the wrong shape")
    if model.temporal_compressor.dL_dweights is None or model.temporal_expander.dL_dweights is None:
        raise AssertionError("temporal projection layers did not receive gradients")
    if model.gradient_clip_value != DEFAULT_GRADIENT_CLIP_VALUE:
        raise AssertionError("smoke-test model did not retain gradient clipping")
    if compressor_weights == model.temporal_compressor.weights:
        raise AssertionError("smoke-test did not apply a temporal-compressor update")
    if not np.isfinite(decoded_array).all():
        raise AssertionError("smoke-test reconstruction contains non-finite values")

    result = {
        "batch_shape": tuple(batch.shape),
        "bottleneck_shape": (2, 16, 4),
        "reconstruction_shape": tuple(reconstruction.shape),
        "gradient_clipping": model.gradient_clip_value,
        "weight_update_applied": True,
    }
    logging.info("SMD smoke test passed: %s", result)
    return result


def run_smoke_tests():
    """Run the required small and full-width temporal model smoke tests."""
    return [run_smoke_test(5), run_smoke_test(38)]


def run_experiment(args):
    """Train, calibrate, evaluate, and save one complete SMD experiment."""
    training, testing, point_labels, paths = load_smd_machine(args.smd_root, args.machine)
    available_feature_count = training.shape[1]
    training, testing = select_features(training, testing, args.feature_count)
    raw_training, raw_validation = split_training_timeline(training, args.train_split)
    scaler_mean, scaler_scale = fit_scaler(raw_training)
    scaled_training = transform(raw_training, scaler_mean, scaler_scale)
    scaled_validation = transform(raw_validation, scaler_mean, scaler_scale)
    scaled_testing = transform(testing, scaler_mean, scaler_scale)

    training_windows = create_window_view(
        scaled_training,
        args.window_size,
        args.step_size,
    )
    validation_windows = create_window_view(
        scaled_validation,
        args.window_size,
        args.step_size,
    )
    testing_windows = create_window_view(
        scaled_testing,
        args.window_size,
        args.step_size,
    )
    test_records = create_test_window_records(
        point_labels,
        args.window_size,
        args.step_size,
    )
    if len(test_records) != len(testing_windows):
        raise ValueError("test labels and generated test windows do not align")

    overlap_percent = max(0.0, (1 - args.step_size / args.window_size) * 100)
    split_summary = {
        "raw_training_timesteps": int(len(raw_training)),
        "raw_validation_timesteps": int(len(raw_validation)),
        "raw_test_timesteps": int(len(testing)),
        "feature_count": int(args.feature_count),
        "available_feature_count": int(available_feature_count),
        "training_windows": int(len(training_windows)),
        "validation_windows": int(len(validation_windows)),
        "test_windows": int(len(testing_windows)),
        "window_shape": list(training_windows.shape[1:]),
        "consecutive_window_overlap_percent": overlap_percent,
    }
    logging.info("SMD data split: %s", split_summary)

    model = (
        load_smd_checkpoint(args.load_checkpoint, args)
        if args.load_checkpoint is not None
        else build_model(args, args.feature_count)
    )
    best_model, history, best_validation_loss = train_model(
        model,
        training_windows,
        validation_windows,
        args.epochs,
    )
    feature_error_normalizer = None
    if args.normalize_feature_errors:
        feature_error_normalizer = fit_feature_error_normalizer(
            best_model,
            validation_windows,
        )
        logging.info("Fitted validation feature-error median/MAD normalizer")
    test_scores = score_windows(
        best_model,
        testing_windows,
        args.score_method,
        args.top_k_timesteps,
        args.top_k_features,
        feature_error_normalizer,
    )
    actual = np.asarray([record["actual_anomaly"] for record in test_records], dtype=int)
    if args.threshold_mode == "static":
        validation_scores = score_windows(
            best_model,
            validation_windows,
            args.score_method,
            args.top_k_timesteps,
            args.top_k_features,
            feature_error_normalizer,
        )
        # Test labels are deliberately excluded from this threshold calculation.
        static_threshold = float(
            np.percentile(validation_scores, args.threshold_percentile)
        )
        thresholds = np.full(len(test_scores), static_threshold, dtype=np.float64)
        predicted = (test_scores >= static_threshold).astype(np.int64)
        calibration_windows = np.zeros(len(test_scores), dtype=bool)
    else:
        thresholds, predicted, calibration_windows = rolling_mad_predictions(
            test_scores,
            args.warmup_windows,
            args.rolling_history_windows,
            args.mad_multiplier,
        )

    evaluation_windows = ~calibration_windows
    evaluated_actual = actual[evaluation_windows]
    evaluated_predicted = predicted[evaluation_windows]
    evaluated_scores = test_scores[evaluation_windows]
    metrics = safe_binary_metrics(
        evaluated_actual,
        evaluated_predicted,
        evaluated_scores,
    )
    metrics.update(
        {
            "mean_normal_window_reconstruction_error": (
                float(np.mean(evaluated_scores[evaluated_actual == 0]))
                if np.any(evaluated_actual == 0)
                else None
            ),
            "mean_anomalous_window_reconstruction_error": (
                float(np.mean(evaluated_scores[evaluated_actual == 1]))
                if np.any(evaluated_actual == 1)
                else None
            ),
            "threshold": float(thresholds[-1]),
            "initial_threshold": (
                float(thresholds[np.flatnonzero(evaluation_windows)[0]])
            ),
            "final_threshold": float(thresholds[-1]),
            "threshold_mode": args.threshold_mode,
            "calibration_windows": int(np.sum(calibration_windows)),
            "evaluation_windows": int(np.sum(evaluation_windows)),
            "total_predicted_anomalies": int(np.sum(evaluated_predicted)),
            "total_actual_anomalous_windows": int(np.sum(evaluated_actual)),
            "best_validation_loss": best_validation_loss,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "smd_root": str(args.smd_root),
        "output_dir": str(args.output_dir),
        "load_checkpoint": (
            str(args.load_checkpoint) if args.load_checkpoint is not None else None
        ),
        "input_size": args.feature_count,
        "gradient_clip_value": DEFAULT_GRADIENT_CLIP_VALUE,
        "dataset": "SMD Server Machine Dataset via OmniAnomaly",
        "machine_paths": {name: str(path) for name, path in paths.items()},
    }
    config.pop("smoke_only", None)
    save_model(
        args.output_dir / "best_model.pt",
        best_model,
        {**config, "dataset": "smd", "machine": args.machine},
    )
    write_json(args.output_dir / "experiment_config.json", config)
    write_json(args.output_dir / "data_split_summary.json", split_summary)
    write_json(
        args.output_dir / "scaler_parameters.json",
        {"mean": scaler_mean.tolist(), "scale": scaler_scale.tolist()},
    )
    if feature_error_normalizer is not None:
        write_json(
            args.output_dir / "feature_error_normalizer.json",
            {
                "median": feature_error_normalizer["median"].tolist(),
                "mad": feature_error_normalizer["mad"].tolist(),
                "robust_scale": feature_error_normalizer["robust_scale"].tolist(),
            },
        )
    write_json(args.output_dir / "metrics.json", metrics)
    write_training_history(args.output_dir / "training_history.csv", history)
    write_window_scores(
        args.output_dir / "window_scores.csv",
        test_records,
        test_scores,
        thresholds,
        predicted,
        calibration_windows,
    )
    save_plots(
        args.output_dir,
        history,
        test_records,
        test_scores,
        thresholds,
        calibration_windows,
    )
    logging.info("SMD metrics: %s", json.dumps(metrics, indent=2, sort_keys=True))
    return split_summary, metrics


def main():
    """Run the SMD command-line workflow or its smoke-test mode."""
    configure_logging()
    args = build_parser().parse_args()
    try:
        validate_arguments(args)
        if args.smoke_only:
            run_smoke_tests()
        else:
            run_experiment(args)
    except Exception as error:
        logging.error("SMD experiment failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
