"""
Train the custom autoencoder from numeric columns in a CSV file.

Example:
    python -m Model.Run data.csv autoencoder.pt --bottleneck 8 --max-rows 5000

The saved .pt file is a Python pickle for this custom model, not a PyTorch
checkpoint. Only load model files from sources you trust.
"""

import argparse
import csv
import logging
import math
import os
import pickle
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from Model.Autoencoder import (
    Autoencoder,
    STEP_SIZE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    WINDOW_SIZE,
)


DEFAULT_MAX_ROWS = 1_000
DEFAULT_MAX_FEATURES = 64
DEFAULT_MAX_PARAMETERS = 1_000_000
DEFAULT_CLIP_VALUE = 10.0
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_GRADIENT_CLIP_VALUE = 1.0
MAX_EPOCHS = 100


def build_parser():
    """Create command-line options for general numeric CSV training."""
    parser = argparse.ArgumentParser(
        description="Train the custom autoencoder on numeric CSV columns."
    )
    parser.add_argument("input_file", type=Path, help="Input .csv file")
    parser.add_argument("output_file", type=Path, help="Output custom .pt model")
    parser.add_argument(
        "--bottleneck",
        "--latent-size",
        dest="bottleneck",
        type=int,
        required=True,
        help="Latent feature width at each timestep.",
    )
    parser.add_argument(
        "--bottleneck-steps",
        type=int,
        help="Compress each input window to this many temporal steps.",
    )
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=STEP_SIZE)
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--test-split", type=float, default=TEST_SPLIT)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--max-parameters", type=int, default=DEFAULT_MAX_PARAMETERS)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--clip-value", type=float, default=DEFAULT_CLIP_VALUE)
    parser.add_argument(
        "--gradient-clip-value",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_VALUE,
    )
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def configure_logging(log_file):
    """Configure console and optional file logging for one training run."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def validate_arguments(args):
    """Reject unsafe files, dimensions, and training settings before loading data."""
    if args.input_file.suffix.lower() != ".csv":
        raise ValueError("input_file must use the .csv extension")
    if args.output_file.suffix.lower() != ".pt":
        raise ValueError("output_file must use the .pt extension")
    if not args.input_file.is_file():
        raise ValueError(f"input file does not exist: {args.input_file}")
    if args.input_file.stat().st_size > args.max_file_bytes:
        raise ValueError(
            f"input file exceeds --max-file-bytes ({args.max_file_bytes:,} bytes)"
        )
    if args.output_file.exists() and not args.overwrite:
        raise ValueError(
            f"output file already exists: {args.output_file}; use --overwrite to replace it"
        )
    if args.input_file.resolve() == args.output_file.resolve():
        raise ValueError("input and output files must be different")
    if args.bottleneck <= 0:
        raise ValueError("bottleneck must be greater than zero")
    if args.bottleneck_steps is not None and (
        args.bottleneck_steps <= 0 or args.bottleneck_steps > args.window_size
    ):
        raise ValueError(
            "bottleneck-steps must be greater than zero and no larger than window-size"
        )
    if not 0 < args.learning_rate <= 1:
        raise ValueError("learning-rate must be greater than zero and no more than one")
    for argument_name in [
        "max_epochs",
        "target",
        "window_size",
        "step_size",
        "max_rows",
        "max_features",
        "max_parameters",
        "max_file_bytes",
    ]:
        if getattr(args, argument_name) <= 0:
            raise ValueError(f"{argument_name.replace('_', '-')} must be greater than zero")
    if not math.isfinite(args.clip_value) or args.clip_value <= 0:
        raise ValueError("clip-value must be a finite number greater than zero")
    if (
        not math.isfinite(args.gradient_clip_value)
        or args.gradient_clip_value <= 0
    ):
        raise ValueError("gradient-clip-value must be a finite number greater than zero")
    if args.max_epochs > MAX_EPOCHS:
        raise ValueError(f"max-epochs cannot exceed {MAX_EPOCHS}")
    Autoencoder.validate_window_settings(
        args.window_size,
        args.step_size,
        args.train_split,
        args.test_split,
    )
    if args.bidirectional and args.window_size < 2:
        raise ValueError(
            "bidirectional training requires --window-size of at least two"
        )


def numeric_columns(input_file):
    """
    Finds columns whose non-empty values can all be converted to floats.
    """
    with input_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV file must include a header row")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV header names must be unique")

        candidate_columns = set(reader.fieldnames)
        numeric_value_counts = {column: 0 for column in reader.fieldnames}
        for row in reader:
            for column in tuple(candidate_columns):
                value = row.get(column)
                if value is None or not value.strip():
                    continue
                try:
                    float(value)
                except ValueError:
                    candidate_columns.remove(column)
                else:
                    numeric_value_counts[column] += 1

    return [
        column
        for column in reader.fieldnames
        if column in candidate_columns and numeric_value_counts[column] > 0
    ]


def estimate_parameter_count(
    feature_count,
    latent_size,
    bidirectional=False,
    window_size=None,
    bottleneck_steps=None,
):
    """Estimate trainable parameters for resource-limit validation."""
    encoder_directions = 2 if bidirectional else 1
    encoder_lstm = encoder_directions * 4 * feature_count * (feature_count + 2)
    linear_encoder_input = feature_count * encoder_directions
    linear_encoder = latent_size * (linear_encoder_input + 2)
    linear_decoder = linear_encoder_input * (latent_size + 2)
    decoder_lstm = 4 * feature_count * (feature_count + 2)
    temporal_projection = 0
    if bottleneck_steps is not None:
        temporal_projection = 2 * window_size * bottleneck_steps
    return (
        encoder_lstm
        + linear_encoder
        + linear_decoder
        + decoder_lstm
        + temporal_projection
    )


def row_to_vector(row, columns, clip_value):
    """Convert one CSV row to a finite, optionally clipped feature vector."""
    vector = []
    clipped_values = 0

    for column in columns:
        value = row[column]
        if value is None or not value.strip():
            return None, clipped_values

        try:
            numeric_value = float(value)
        except ValueError:
            return None, clipped_values
        if not math.isfinite(numeric_value):
            return None, clipped_values
        if numeric_value > clip_value:
            numeric_value = clip_value
            clipped_values += 1
        elif numeric_value < -clip_value:
            numeric_value = -clip_value
            clipped_values += 1
        vector.append(numeric_value)

    return vector, clipped_values


def save_model(output_file, model, metadata):
    """Atomically save a custom-model pickle and its metadata."""
    payload = {
        "format": "custom-autoencoder-pickle-v1",
        "model": model,
        "metadata": metadata,
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.",
        suffix=".tmp",
        dir=output_file.parent,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            pickle.dump(payload, temporary_file, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, output_file)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise


def load_model(input_file, expected_bottleneck_steps=None):
    """Loads a saved custom model and rejects incompatible temporal checkpoints."""
    with input_file.open("rb") as model_file:
        payload = pickle.load(model_file)

    if payload.get("format") != "custom-autoencoder-pickle-v1":
        raise ValueError("unsupported custom autoencoder checkpoint format")

    model = payload.get("model")
    actual_steps = getattr(model, "bottleneck_steps", None)
    if expected_bottleneck_steps is not None and actual_steps != expected_bottleneck_steps:
        raise ValueError(
            "checkpoint temporal bottleneck is incompatible: expected "
            f"{expected_bottleneck_steps} steps, found {actual_steps!r}. "
            "Older checkpoints without bottleneck_steps cannot be loaded as "
            "temporal-bottleneck models."
        )

    for layer_name in [
        "encoder",
        "forward_encoder",
        "backward_encoder",
        "decoder",
    ]:
        layer = getattr(model, layer_name, None)
        if layer is None:
            continue
        expected_gate_width = layer.input_size + layer.output_size
        actual_gate_width = len(layer.nodes[0].cells[0].weights)
        if actual_gate_width != expected_gate_width:
            raise ValueError(
                f"checkpoint {layer_name} uses legacy scalar-LSTM gate rows of "
                f"width {actual_gate_width}; this vector-LSTM model requires "
                f"width {expected_gate_width}. Retrain the checkpoint."
            )
    return payload


def train(args):
    """Load a CSV, create chronological windows, train, and save the model."""
    # Inspect columns before constructing the model so resource limits can be
    # enforced from the actual input shape.
    columns = numeric_columns(args.input_file)
    if not columns:
        raise ValueError("the CSV file contains no numeric columns")
    if len(columns) > args.max_features:
        raise ValueError(
            f"found {len(columns)} numeric columns, which exceeds --max-features "
            f"({args.max_features})"
        )

    parameter_count = estimate_parameter_count(
        len(columns),
        args.bottleneck,
        args.bidirectional,
        args.window_size,
        args.bottleneck_steps,
    )
    if parameter_count > args.max_parameters:
        raise ValueError(
            f"estimated parameter count ({parameter_count:,}) exceeds --max-parameters "
            f"({args.max_parameters:,})"
        )

    model = Autoencoder(
        len(columns),
        args.learning_rate,
        args.bottleneck,
        max_epochs=args.max_epochs,
        bidirectional=args.bidirectional,
        gradient_clip_value=args.gradient_clip_value,
        bottleneck_steps=args.bottleneck_steps,
        window_size=args.window_size if args.bottleneck_steps is not None else None,
    )
    logging.info(
        "Model construction: features=%d latent_size=%d bottleneck_steps=%s "
        "learning_rate=%s max_epochs=%d estimated_parameters=%d bidirectional=%s",
        len(columns),
        args.bottleneck,
        args.bottleneck_steps,
        args.learning_rate,
        args.max_epochs,
        parameter_count,
        model.bidirectional,
    )
    logging.info(
        "Safety limits: input_bytes=%d max_file_bytes=%d max_rows=%d "
        "max_features=%d max_parameters=%d clip_value=%s "
        "gradient_clip_value=%s",
        args.input_file.stat().st_size,
        args.max_file_bytes,
        args.max_rows,
        args.max_features,
        args.max_parameters,
        args.clip_value,
        args.gradient_clip_value,
    )
    logging.info("Using numeric columns: %s", ", ".join(columns))
    logging.info(
        "Window settings: window_size=%d step_size=%d train_split=%.2f "
        "test_split=%.2f",
        args.window_size,
        args.step_size,
        args.train_split,
        args.test_split,
    )

    rows = []
    skipped_rows = 0
    clipped_values = 0
    with args.input_file.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if len(rows) >= args.max_rows:
                break

            vector, row_clipped_values = row_to_vector(
                row,
                columns,
                args.clip_value,
            )
            clipped_values += row_clipped_values
            if vector is None:
                skipped_rows += 1
            rows.append(vector)

    training_rows, testing_rows, training_windows, testing_windows = (
        Autoencoder.prepare_windowed_splits(
            rows,
            args.window_size,
            args.step_size,
            args.train_split,
            args.test_split,
        )
    )
    window_shape = (len(training_windows[0]), len(training_windows[0][0]))
    logging.info(
        "Dataset split: training_rows=%d testing_rows=%d training_windows=%d "
        "testing_windows=%d window_shape=%s",
        len(training_rows),
        len(testing_rows),
        len(training_windows),
        len(testing_windows),
        window_shape,
    )

    for window_index, training_window in enumerate(training_windows, start=1):
        model.run(training_window, args.target)
        if window_index % 100 == 0:
            logging.info(
                "Training progress: windows=%d skipped_rows=%d clipped_values=%d",
                window_index,
                skipped_rows,
                clipped_values,
            )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.input_file.resolve()),
        "feature_columns": columns,
        "feature_count": len(columns),
        "bottleneck": args.bottleneck,
        "latent_size": args.bottleneck,
        "bottleneck_steps": args.bottleneck_steps,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "bidirectional": args.bidirectional,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "gradient_clip_value": args.gradient_clip_value,
        "training_rows": len(training_rows),
        "testing_rows": len(testing_rows),
        "training_windows": len(training_windows),
        "testing_windows": len(testing_windows),
        "window_shape": window_shape,
        "skipped_rows": skipped_rows,
        "clipped_values": clipped_values,
        "estimated_parameter_count": parameter_count,
    }
    save_model(args.output_file, model, metadata)
    logging.info(
        "Saved model: %s (training_windows=%d testing_windows=%d skipped_rows=%d "
        "clipped_values=%d)",
        args.output_file,
        len(training_windows),
        len(testing_windows),
        skipped_rows,
        clipped_values,
    )


def main():
    """Run the command-line CSV training workflow and report failures."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)
        if args.log_file is None:
            args.log_file = args.output_file.with_suffix(args.output_file.suffix + ".log")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        configure_logging(args.log_file)
        train(args)
    except KeyboardInterrupt:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        logging.warning("Training interrupted; no model was saved.")
        return 130
    except Exception as error:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        logging.error("Training failed: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
