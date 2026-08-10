# Custom LSTM Autoencoder

This project is a from scratch Python implementation of a sequence
autoencoder for anomaly detection. It is intentionally built without a deep
learning framework: the LSTM gates, linear layers, temporal bottleneck,
backpropagation, gradient clipping, and parameter updates are implemented in
the project source.

The main current experiment uses the Server Machine Dataset (SMD) to learn
normal multivariate server behavior and flag unusual reconstruction error.

## Why This Project Exists

The research lab I am part of at the University of Georgia (UGA) uses a GAN
model whose generator is an autoencoder. I built this project to understand the
math behind that work rather than treating the model as a black box. Implementing
the encoder, decoder, LSTM gates, gradients, and parameter updates by hand has
been a way to study why an autoencoder can learn a normal pattern, how it
reconstructs an input, and how reconstruction error can become an anomaly
signal.

My [learning notes](Notes/) contain the handwritten material and working notes
from that study process.

## Model

The temporal autoencoder reconstructs an input window in this form:

```text
[window_size, input_features]
  -> vector LSTM encoder
  -> temporal compression
  -> latent linear projection
  -> latent linear expansion
  -> temporal expansion
  -> vector LSTM decoder
  -> [window_size, input_features]
```

For the default SMD experiment, the shape flow is:

```text
[128, 38] -> [16, 16] -> [16, 8] -> [128, 38]
```

`bottleneck_steps` controls the compressed number of timesteps, while
`latent_size` controls the number of features retained at each compressed
timestep. They are separate settings.

The LSTM is vector-based: each gate receives the current input vector and the
entire previous hidden-state vector. The project caches the forward-pass values
needed for backpropagation through time, accumulates gate gradients across a
sequence, and updates gate weights and biases with the configured learning
rate. Xavier initialization and optional per-gradient clipping are enabled for
the temporal model.

## Project Layout

- `Model/Autoencoder.py`: sequence autoencoder, temporal compression/expansion, loss,
  and full model backpropagation.
- `Model/Networks.py`: vector LSTM, linear layers, temporal projections, gradient
  calculations, and parameter updates.
- `Model/Node.py`: low-level node and gate-cell implementation.
- `Model/Run.py`: general CSV trainer and custom model save/load helpers.
- `Model/run_smd_experiment.py`: SMD-specific training, validation, scoring,
  thresholding, metrics, plots, and output files.
- `Test/test_*.py`: numerical, windowing, temporal-bottleneck, vector-LSTM, and SMD
  tests.

## SMD Experiment

The SMD runner loads one machine at a time from the OmniAnomaly repository.
Training data is split chronologically before windows are generated: the first
portion fits the scaler and trains the model; the final portion validates it.
Test labels are never used for scaling or threshold selection.

Each test-window score first selects the highest reconstruction-error sensors at
each timestep, then selects the highest timestep scores in the window. Use
`--top-k-features` and `--top-k-timesteps` to control those two selections.
By default, an adaptive rolling-MAD threshold uses an initial assumed-normal
warm-up period and then updates from recent unflagged scores. Warm-up windows
are excluded from metrics. Use `--threshold-mode static` to retain the fixed
validation-percentile threshold.

Run the two architecture and backpropagation smoke tests:

```bash
python3 -u -m Model.run_smd_experiment --smoke-only
```

Run the recommended full SMD baseline:

```bash
python3 -u -m Model.run_smd_experiment \
  --smd-root data/OmniAnomaly/ServerMachineDataset \
  --machine machine-1-1 \
  --window-size 128 \
  --step-size 16 \
  --hidden-size 24 \
  --latent-size 8 \
  --bottleneck-steps 12 \
  --epochs 20 \
  --learning-rate 0.0005 \
  --train-split 0.80 \
  --threshold-mode rolling_mad \
  --warmup-windows 256 \
  --rolling-history-windows 512 \
  --mad-multiplier 6 \
  --threshold-percentile 99 \
  --feature-count 38 \
  --score-method top_k \
  --top-k-features 3 \
  --top-k-timesteps 8 \
  --seed 42 \
  --output-dir outputs/smd_machine_1_1_h24_l8_b12_top3_top8_rolling_mad
```

The runner writes the best model, scaler parameters, training history,
window-level scores, metrics, split summary, configuration, and plots to the
selected output directory. Custom `.pt` files are Python pickle files, not
PyTorch checkpoints. Only load checkpoints you trust.

Compare completed SMD experiment folders in one Matplotlib figure:

```bash
python3 -m Model.plot_smd_results
```

The comparison is saved to `outputs/smd_experiment_comparison.png`. Pass
`--experiment-dir` more than once to plot only specific runs.

## General CSV Training

`Model/Run.py` trains the same model on numeric columns in a CSV file and stores a
custom checkpoint:

```bash
python3 -m Model.Run input.csv model.pt --bottleneck 8 --bottleneck-steps 16
```

Use its limits for large files, feature counts, parameter counts, value
clipping, and gradient clipping when working with unfamiliar data.

## Dataset Attribution

The SMD experiment uses the Server Machine Dataset released with the
[OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) project. The dataset
repository is downloaded locally under `data/OmniAnomaly/`, is ignored by Git,
and is not part of this project’s model implementation.
