"""Sequence autoencoder orchestration and temporal-window utilities."""

from Model.Networks import Lstm, Linear, TemporalProjection
from sympy import symbols, diff
import math

# Dataset windowing configuration
STEP_SIZE = 3
WINDOW_SIZE = 36
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.20

class Autoencoder:
    """Represents the full LSTM reconstruction model in the ML library.

    The autoencoder sends input through an encoder, optional temporal
    compression, latent feature projections, temporal expansion, and a decoder.
    It supports individual vectors, sequences, and bidirectional encoders.

    Attributes:
        input_size: Number of input features at each timestep.
        hidden_size: Width of the encoder hidden-state vector.
        latent_size: Feature width retained at each bottleneck timestep.
        bottleneck_steps: Compressed sequence length in temporal mode.
        encoder: Forward encoder for unidirectional models.
        decoder: LSTM that reconstructs the original input features.
    """

    @staticmethod
    def validate_window_settings(window_size, step_size, train_split, test_split):
        """Validate shared chronological-window configuration values."""
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        if step_size <= 0:
            raise ValueError("step_size must be greater than zero")
        if not 0 < train_split < 1:
            raise ValueError("train_split must be between zero and one")
        if not 0 < test_split < 1:
            raise ValueError("test_split must be between zero and one")
        if not math.isclose(train_split + test_split, 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("train_split and test_split must add up to one")

    @staticmethod
    def split_chronologically(rows, train_split=TRAIN_SPLIT, test_split=TEST_SPLIT):
        """Split rows into earlier training and later testing timelines."""
        Autoencoder.validate_window_settings(1, 1, train_split, test_split)
        split_index = int(len(rows) * train_split)
        return rows[:split_index], rows[split_index:]

    @staticmethod
    def create_sliding_windows(rows, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
        """
        Creates complete, same-shaped windows without crossing invalid CSV rows.
        """
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        if step_size <= 0:
            raise ValueError("step_size must be greater than zero")

        windows = []
        feature_count = None
        for start_index in range(0, len(rows) - window_size + 1, step_size):
            window = rows[start_index:start_index + window_size]
            if any(row is None for row in window):
                continue

            if feature_count is None:
                feature_count = len(window[0])
            if any(len(row) != feature_count for row in window):
                raise ValueError("every window row must have the same feature count")

            windows.append([row[:] for row in window])

        return windows

    @staticmethod
    def prepare_windowed_splits(
        rows,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        train_split=TRAIN_SPLIT,
        test_split=TEST_SPLIT,
    ):
        """
        Splits chronological rows before creating independent train and test windows.
        """
        Autoencoder.validate_window_settings(
            window_size,
            step_size,
            train_split,
            test_split,
        )
        training_rows, testing_rows = Autoencoder.split_chronologically(
            rows,
            train_split,
            test_split,
        )

        if len(training_rows) < window_size:
            raise ValueError("training split does not contain one complete window")
        if len(testing_rows) < window_size:
            raise ValueError("testing split does not contain one complete window")

        training_windows = Autoencoder.create_sliding_windows(
            training_rows,
            window_size,
            step_size,
        )
        testing_windows = Autoencoder.create_sliding_windows(
            testing_rows,
            window_size,
            step_size,
        )

        if not training_windows:
            raise ValueError("training split has no complete valid windows")
        if not testing_windows:
            raise ValueError("testing split has no complete valid windows")

        return training_rows, testing_rows, training_windows, testing_windows

    def __init__(self, input_size, learning_rate, bottleneck=None, loss_type="mse", target_type="early_stopping", target=10, bidirectional=False, max_epochs=None, gradient_clip_value=None, bottleneck_steps=None, latent_size=None, hidden_size=None, window_size=None):
        """Create a configurable LSTM autoencoder and its trainable layers."""
        self.input_size = input_size
        self.learning_rate = learning_rate
        if latent_size is None:
            latent_size = bottleneck
        if latent_size is None or latent_size <= 0:
            raise ValueError("latent_size or legacy bottleneck must be greater than zero")
        self.latent_size = latent_size
        # Retained for callers that still read the old public attribute.
        self.bottleneck = latent_size
        self.bottleneck_steps = bottleneck_steps
        self.window_size = window_size
        self.temporal_bottleneck = bottleneck_steps is not None
        if self.temporal_bottleneck:
            if window_size is None or window_size <= 0:
                raise ValueError("window_size is required with bottleneck_steps")
            if bottleneck_steps <= 0 or bottleneck_steps > window_size:
                raise ValueError(
                    "bottleneck_steps must be greater than zero and no larger than window_size"
                )
        self.bidirectional = bidirectional
        self.target_type = target_type
        self.target = target
        self.max_epochs = max_epochs
        self.loss_type = loss_type
        if gradient_clip_value is not None and (
            not math.isfinite(gradient_clip_value) or gradient_clip_value <= 0
        ):
            raise ValueError("gradient_clip_value must be a finite number greater than zero")
        self.gradient_clip_value = gradient_clip_value

        self.hidden_size = input_size if hidden_size is None else hidden_size
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")

        # The temporal projections change sequence length; linear layers only
        # change feature width at each retained timestep.
        if self.bidirectional:
            self.forward_encoder = Lstm(input_size, learning_rate, self.hidden_size)
            self.backward_encoder = Lstm(input_size, learning_rate, self.hidden_size)
            encoder_output_size = self.forward_encoder.output_size
            encoder_sequence_size = encoder_output_size * 2
        else:
            self.encoder = Lstm(input_size, learning_rate, self.hidden_size)
            encoder_output_size = self.encoder.output_size
            encoder_sequence_size = encoder_output_size

        self.encoder_output_size = encoder_output_size
        self.encoder_sequence_size = encoder_sequence_size
        self.linear_encoder = Linear(
            encoder_sequence_size,
            learning_rate,
            self.latent_size,
        )
        self.linear_decoder = Linear(
            self.latent_size,
            learning_rate,
            encoder_sequence_size,
        )
        self.decoder = Lstm(encoder_sequence_size, learning_rate, input_size)
        if self.temporal_bottleneck:
            self.temporal_compressor = TemporalProjection(
                window_size,
                bottleneck_steps,
                encoder_sequence_size,
                learning_rate,
            )
            self.temporal_expander = TemporalProjection(
                bottleneck_steps,
                window_size,
                encoder_sequence_size,
                learning_rate,
            )
        self.set_gradient_clipping()

    def run(self, input, target, reconstruction_target=None):
        """Train on one vector or dispatch a sequence to the matching path."""
        if self.is_sequence(input):
            return self.run_sequence(input, target, reconstruction_target)

        if self.temporal_bottleneck:
            raise ValueError("temporal bottleneck models require a full input sequence")

        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")
        if reconstruction_target is None:
            reconstruction_target = input
        if len(reconstruction_target) != self.input_size:
            raise ValueError("reconstruction target must match the input size configuration")

        validation = list()

        stopper = False

        while (not stopper):
            self.reset_network_state()

            if self.bidirectional:
                forward_encoder_output = self.forward_encoder.run(input)
                backward_encoder_output = self.backward_encoder.run(input)
                encoded = self.combine_encoder_outputs(
                    forward_encoder_output,
                    backward_encoder_output,
                )
            else:
                encoded = self.encoder.run(input)

            bottleneck_output = self.linear_encoder.run(encoded)
            bottleneck_input = self.linear_decoder.run(bottleneck_output)
            decoded = self.decoder.run(bottleneck_input)

            loss_list = self.compute_loss(decoded, reconstruction_target)
            loss_mean = sum(loss_list) / len(loss_list)
            validation.append(loss_mean)

            self.backpropegate(reconstruction_target, decoded)
            if self.check_stop(validation, target):
                stopper = True

        return decoded

    def run_sequence(self, sequence, target, reconstruction_target=None):
        """
        Reconstructs a sequence of feature vectors and trains on every timestep.
        """
        self.validate_sequence(sequence)
        if reconstruction_target is None:
            reconstruction_target = sequence
        self.validate_sequence(reconstruction_target)
        if len(reconstruction_target) != len(sequence):
            raise ValueError("reconstruction target must match the input sequence length")

        if self.temporal_bottleneck:
            return self.run_temporal_sequence(
                sequence,
                target,
                reconstruction_target,
            )

        validation = []
        stopper = False

        while not stopper:
            self.reset_network_state()

            if self.bidirectional:
                forward_outputs = [
                    self.forward_encoder.run(timestep)
                    for timestep in sequence
                ]
                backward_outputs_reversed = [
                    self.backward_encoder.run(timestep)
                    for timestep in reversed(sequence)
                ]
                backward_outputs = list(reversed(backward_outputs_reversed))
                encoded_outputs = [
                    self.combine_encoder_outputs(
                        forward_output,
                        backward_output,
                    )
                    for forward_output, backward_output in zip(
                        forward_outputs,
                        backward_outputs,
                    )
                ]
            else:
                encoded_outputs = [
                    self.encoder.run(timestep)
                    for timestep in sequence
                ]

            bottleneck_outputs = [
                self.linear_encoder.run(encoded)
                for encoded in encoded_outputs
            ]
            decoder_inputs = [
                self.linear_decoder.run(bottleneck_output)
                for bottleneck_output in bottleneck_outputs
            ]
            decoded = [
                self.decoder.run(decoder_input)
                for decoder_input in decoder_inputs
            ]

            loss_list = [
                loss
                for decoded_timestep, input_timestep in zip(
                    decoded,
                    reconstruction_target,
                )
                for loss in self.compute_loss(decoded_timestep, input_timestep)
            ]
            validation.append(sum(loss_list) / len(loss_list))

            self.backpropegate(reconstruction_target, decoded)
            if self.check_stop(validation, target):
                stopper = True

        return decoded

    def run_temporal_sequence(self, sequence, target, reconstruction_target=None):
        """Train a fixed-length sequence through the temporal bottleneck."""
        self.validate_temporal_sequence(sequence)
        if reconstruction_target is None:
            reconstruction_target = sequence
        self.validate_temporal_sequence(reconstruction_target)
        if len(reconstruction_target) != len(sequence):
            raise ValueError("reconstruction target must match the input sequence length")
        validation = []
        stopper = False

        while not stopper:
            decoded = self.forward_temporal_sequence(sequence)
            self.assert_reconstruction_shape(reconstruction_target, decoded)
            loss_list = [
                loss
                for decoded_timestep, input_timestep in zip(
                    decoded,
                    reconstruction_target,
                )
                for loss in self.compute_loss(decoded_timestep, input_timestep)
            ]
            validation.append(sum(loss_list) / len(loss_list))
            self.backpropegate_temporal_sequence(reconstruction_target, decoded)
            if self.check_stop(validation, target):
                stopper = True

        return decoded

    def forward_temporal_sequence(self, sequence):
        """Reconstruct one temporal window and cache layer outputs for gradients."""
        self.validate_temporal_sequence(sequence)
        # Windows are independent examples, so recurrent state must not leak
        # from the preceding window into this forward pass.
        self.reset_network_state()
        encoded_outputs = self.encode_sequence(sequence)
        compressed_outputs = self.temporal_compressor.run(encoded_outputs)
        latent_outputs = [
            self.linear_encoder.run(compressed_output)
            for compressed_output in compressed_outputs
        ]
        decoded_latent_outputs = [
            self.linear_decoder.run(latent_output)
            for latent_output in latent_outputs
        ]
        expanded_outputs = self.temporal_expander.run(decoded_latent_outputs)
        decoded = [self.decoder.run(expanded_output) for expanded_output in expanded_outputs]
        self.last_temporal_shapes = {
            "input": (len(sequence), len(sequence[0])),
            "encoder": (len(encoded_outputs), len(encoded_outputs[0])),
            "compressed": (len(compressed_outputs), len(compressed_outputs[0])),
            "bottleneck": (len(latent_outputs), len(latent_outputs[0])),
            "expanded": (len(expanded_outputs), len(expanded_outputs[0])),
            "reconstruction": (len(decoded), len(decoded[0])),
        }
        return decoded

    def encode_sequence(self, sequence):
        """Return encoder outputs in original timestep order."""
        if self.bidirectional:
            forward_outputs = [
                self.forward_encoder.run(timestep)
                for timestep in sequence
            ]
            backward_outputs_reversed = [
                self.backward_encoder.run(timestep)
                for timestep in reversed(sequence)
            ]
            backward_outputs = list(reversed(backward_outputs_reversed))
            return [
                self.combine_encoder_outputs(forward_output, backward_output)
                for forward_output, backward_output in zip(
                    forward_outputs,
                    backward_outputs,
                )
            ]

        return [self.encoder.run(timestep) for timestep in sequence]

    def reconstruct(self, input):
        """
        Runs a forward pass without calculating gradients or updating parameters.
        """
        if self.is_sequence(input):
            if self.temporal_bottleneck:
                decoded = self.forward_temporal_sequence(input)
                self.assert_reconstruction_shape(input, decoded)
                return decoded

            self.validate_sequence(input)
            self.reset_network_state()

            if self.bidirectional:
                forward_outputs = [
                    self.forward_encoder.run(timestep)
                    for timestep in input
                ]
                backward_outputs_reversed = [
                    self.backward_encoder.run(timestep)
                    for timestep in reversed(input)
                ]
                backward_outputs = list(reversed(backward_outputs_reversed))
                encoded_outputs = [
                    self.combine_encoder_outputs(forward_output, backward_output)
                    for forward_output, backward_output in zip(
                        forward_outputs,
                        backward_outputs,
                    )
                ]
            else:
                encoded_outputs = [
                    self.encoder.run(timestep)
                    for timestep in input
                ]

            bottleneck_outputs = [
                self.linear_encoder.run(encoded)
                for encoded in encoded_outputs
            ]
            decoder_inputs = [
                self.linear_decoder.run(bottleneck_output)
                for bottleneck_output in bottleneck_outputs
            ]
            return [self.decoder.run(decoder_input) for decoder_input in decoder_inputs]

        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")

        self.reset_network_state()
        if self.bidirectional:
            encoded = self.combine_encoder_outputs(
                self.forward_encoder.run(input),
                self.backward_encoder.run(input),
            )
        else:
            encoded = self.encoder.run(input)

        return self.decoder.run(
            self.linear_decoder.run(self.linear_encoder.run(encoded))
        )

    def reconstruct_batch(self, batch):
        """Reconstruct independent sequences without updating model parameters."""
        if not batch:
            raise ValueError("batch must contain at least one sequence")

        reconstructed_batch = [self.reconstruct(sequence) for sequence in batch]
        for sequence, reconstructed in zip(batch, reconstructed_batch):
            self.assert_reconstruction_shape(sequence, reconstructed)
        return reconstructed_batch

    def check_stop(self, rounds, target=None):
        """Return whether the configured epoch or early-stopping rule is met."""
        if self.max_epochs is not None and len(rounds) >= self.max_epochs:
            return True

        if target is None:
            target = self.target

        if self.target_type == "early_stopping":
            if len(rounds) <= target:
                return False
            previous_best = min(rounds[:-target])
            recent_best = min(rounds[-target:])
            return recent_best >= previous_best
        else:
            return True

    def compute_loss(self, decoded, input):
        """Return one reconstruction-loss value per output feature."""
        loss_list = list()
        for i in range(len(decoded)):
            if self.loss_type == "mse":
                loss_list.append((decoded[i] - input[i]) ** 2)
        return loss_list

    def backpropegate(self, input, decoded):
        """Backpropagate one vector reconstruction through all trainable layers."""
        if self.is_sequence(input):
            return self.backpropegate_sequence(input, decoded)

        # Gradient of the loss with respect to the decoder input.
        dL_ddecoder_input = self.decoder_propegation(input, decoded)
        dL_dencoder_output = self.linear_propegation(dL_ddecoder_input)
        self.encoder_propegation(dL_dencoder_output)

    def backpropegate_sequence(self, sequence, decoded):
        """
        Backpropagates every reconstruction timestep through the full pipeline.
        """
        if self.temporal_bottleneck:
            return self.backpropegate_temporal_sequence(sequence, decoded)

        if len(decoded) != len(sequence):
            raise ValueError("decoded sequence must match the input sequence length")

        dL_ddecoded = [
            [
                gradient / len(sequence)
                for gradient in self.pd_loss_pd_y(input_timestep, decoded_timestep)
            ]
            for input_timestep, decoded_timestep in zip(sequence, decoded)
        ]
        dL_ddecoder_inputs = self.decoder.backpropegate_sequence(dL_ddecoded)
        dL_dbottleneck_outputs = self.linear_decoder.backpropegate_sequence(
            dL_ddecoder_inputs
        )
        dL_dencoder_outputs = self.linear_encoder.backpropegate_sequence(
            dL_dbottleneck_outputs
        )

        if self.bidirectional:
            expected_size = self.encoder_output_size * 2
            dL_dforward_outputs = []
            dL_dbackward_outputs = []

            for dL_dencoder_output in dL_dencoder_outputs:
                if len(dL_dencoder_output) != expected_size:
                    raise ValueError(
                        "bidirectional encoder gradient must contain "
                        f"{expected_size} values"
                    )

                dL_dforward_output = dL_dencoder_output[:self.encoder_output_size]
                dL_dbackward_output = dL_dencoder_output[self.encoder_output_size:]

                if len(dL_dforward_output) != self.encoder_output_size:
                    raise ValueError("forward encoder gradient has the wrong size")
                if len(dL_dbackward_output) != self.encoder_output_size:
                    raise ValueError("backward encoder gradient has the wrong size")

                dL_dforward_outputs.append(dL_dforward_output)
                dL_dbackward_outputs.append(dL_dbackward_output)

            dL_dforward_inputs = self.forward_encoder.backpropegate_sequence(
                dL_dforward_outputs
            )
            dL_dbackward_inputs = self.backward_encoder.backpropegate_sequence(
                list(reversed(dL_dbackward_outputs))
            )

            return dL_dforward_inputs, list(reversed(dL_dbackward_inputs))

        return self.encoder.backpropegate_sequence(dL_dencoder_outputs)

    def backpropegate_temporal_sequence(self, sequence, decoded):
        """Backpropagate a temporally compressed sequence reconstruction."""
        self.assert_reconstruction_shape(sequence, decoded)
        # Follow the forward pipeline in reverse so each layer receives the
        # gradient with respect to the values it produced.
        dL_ddecoded = [
            [
                gradient / len(sequence)
                for gradient in self.pd_loss_pd_y(input_timestep, decoded_timestep)
            ]
            for input_timestep, decoded_timestep in zip(sequence, decoded)
        ]
        dL_dexpanded_outputs = self.decoder.backpropegate_sequence(dL_ddecoded)
        dL_ddecoded_latent_outputs = self.temporal_expander.backpropegate(
            dL_dexpanded_outputs
        )
        dL_dlatent_outputs = self.linear_decoder.backpropegate_sequence(
            dL_ddecoded_latent_outputs
        )
        dL_dcompressed_outputs = self.linear_encoder.backpropegate_sequence(
            dL_dlatent_outputs
        )
        dL_dencoder_outputs = self.temporal_compressor.backpropegate(
            dL_dcompressed_outputs
        )
        return self.backpropegate_encoder_sequence(dL_dencoder_outputs)

    def backpropegate_encoder_sequence(self, dL_dencoder_outputs):
        """Route encoder gradients into one or both directional LSTMs."""
        if self.bidirectional:
            expected_size = self.encoder_output_size * 2
            dL_dforward_outputs = []
            dL_dbackward_outputs = []

            for dL_dencoder_output in dL_dencoder_outputs:
                if len(dL_dencoder_output) != expected_size:
                    raise ValueError(
                        "bidirectional encoder gradient must contain "
                        f"{expected_size} values"
                    )

                dL_dforward_outputs.append(
                    dL_dencoder_output[:self.encoder_output_size]
                )
                dL_dbackward_outputs.append(
                    dL_dencoder_output[self.encoder_output_size:]
                )

            dL_dforward_inputs = self.forward_encoder.backpropegate_sequence(
                dL_dforward_outputs
            )
            dL_dbackward_inputs = self.backward_encoder.backpropegate_sequence(
                list(reversed(dL_dbackward_outputs))
            )
            return dL_dforward_inputs, list(reversed(dL_dbackward_inputs))

        return self.encoder.backpropegate_sequence(dL_dencoder_outputs)

    def decoder_propegation(self, input, decoded):
        """Differentiate reconstruction loss through the decoder LSTM."""

        dL_ddecoded = self.pd_loss_pd_y(input, decoded)
        dL_ddecoder_input = self.decoder.backpropegate(dL_ddecoded)

        return dL_ddecoder_input

    def linear_propegation(self, dL_ddecoder_input):
        """Pass decoder gradients backward through both linear projections."""

        # Gradient passed from the linear decoder to the bottleneck.
        dL_dbottleneck = self.linear_decoder.backpropegate(dL_ddecoder_input)

        # Gradient passed from the linear encoder to the encoder output.
        dL_dencoder_output = self.linear_encoder.backpropegate(dL_dbottleneck)

        return dL_dencoder_output

    def encoder_propegation(self, dL_dencoder_output):
        """Apply encoder updates for a single vector pass."""

        if self.bidirectional:
            expected_size = self.encoder_output_size * 2
            if len(dL_dencoder_output) != expected_size:
                raise ValueError(
                    "bidirectional encoder gradient must contain "
                    f"{expected_size} values"
                )

            dL_dforward_output = dL_dencoder_output[:self.encoder_output_size]
            dL_dbackward_output = dL_dencoder_output[self.encoder_output_size:]

            if len(dL_dforward_output) != self.encoder_output_size:
                raise ValueError("forward encoder gradient has the wrong size")
            if len(dL_dbackward_output) != self.encoder_output_size:
                raise ValueError("backward encoder gradient has the wrong size")

            dL_dforward_input = self.forward_encoder.backpropegate(
                dL_dforward_output
            )
            dL_dbackward_input = self.backward_encoder.backpropegate(
                dL_dbackward_output
            )

            return dL_dforward_input, dL_dbackward_input

        dL_dencoder_input = self.encoder.backpropegate(dL_dencoder_output)

        return dL_dencoder_input

    def combine_encoder_outputs(self, forward_encoder_output, backward_encoder_output):
        """Concatenate forward and backward hidden vectors safely."""
        if len(forward_encoder_output) != self.encoder_output_size:
            raise ValueError("forward encoder output has the wrong size")
        if len(backward_encoder_output) != self.encoder_output_size:
            raise ValueError("backward encoder output has the wrong size")

        combined_encoder_output = forward_encoder_output + backward_encoder_output
        expected_size = self.encoder_output_size * 2

        if len(combined_encoder_output) != expected_size:
            raise ValueError("combined encoder output has the wrong size")
        if len(combined_encoder_output) != self.linear_encoder.input_size:
            raise ValueError(
                "combined encoder output must match the linear encoder input size"
            )

        return combined_encoder_output

    def is_sequence(self, input):
        """Return whether input is a list of timestep vectors."""
        return bool(input) and isinstance(input[0], list)

    def validate_sequence(self, sequence):
        """Ensure every timestep has the configured feature width."""
        if not sequence:
            raise ValueError("sequence must contain at least one timestep")

        for timestep in sequence:
            if not isinstance(timestep, list):
                raise ValueError("every sequence timestep must be a list")
            if len(timestep) != self.input_size:
                raise ValueError(
                    "every sequence timestep must match the input size configuration"
                )

    def validate_temporal_sequence(self, sequence):
        """Ensure a sequence also matches the configured temporal length."""
        self.validate_sequence(sequence)
        if len(sequence) != self.window_size:
            raise ValueError(
                "temporal bottleneck sequences must contain exactly "
                f"{self.window_size} timesteps"
            )

    def assert_reconstruction_shape(self, input_sequence, reconstructed_sequence):
        """Ensure reconstruction preserves timestep and feature dimensions."""
        if len(reconstructed_sequence) != len(input_sequence):
            raise ValueError(
                "reconstruction timestep count must match the input sequence"
            )

        for timestep_index, (input_timestep, reconstructed_timestep) in enumerate(
            zip(input_sequence, reconstructed_sequence)
        ):
            if len(reconstructed_timestep) != len(input_timestep):
                raise ValueError(
                    "reconstruction feature count must match the input at timestep "
                    f"{timestep_index}"
                )

    def reset_network_state(self):
        """
        Starts an independent training pass with clean recurrent state and caches.
        """
        if self.bidirectional:
            self.forward_encoder.reset_state()
            self.backward_encoder.reset_state()
        else:
            self.encoder.reset_state()

        self.decoder.reset_state()
        self.linear_encoder.reset_cache()
        self.linear_decoder.reset_cache()

    def set_gradient_clipping(self):
        """Apply this model's clipping limit to every trainable sublayer."""
        networks = [self.linear_encoder, self.linear_decoder, self.decoder]
        if self.temporal_bottleneck:
            networks.extend([self.temporal_compressor, self.temporal_expander])
        if self.bidirectional:
            networks.extend([self.forward_encoder, self.backward_encoder])
        else:
            networks.append(self.encoder)

        for network in networks:
            network.gradient_clip_value = self.gradient_clip_value


    def pd_loss_pd_y(self, actual, predicted):
        """Return the MSE derivative with respect to predicted features."""
        if self.loss_type == "mse":
            return [
                2 * (predicted[i] - actual[i])
                / len(actual)
                for i in range(len(actual))
            ]
        else:
            raise NotImplementedError(
                f"Loss type '{self.loss_type}' has not been implemented yet.")

    def pd_y_pd_weights(self, input, previous_hidden=None):
        """Return the legacy scalar-LSTM input used for weight derivatives."""
        if previous_hidden is None:
            previous_hidden = [0]
        elif not isinstance(previous_hidden, list):
            previous_hidden = [previous_hidden]

        if len(previous_hidden) != 1:
            raise ValueError("previous_hidden must contain exactly one value")

        return previous_hidden + input

    def matrix_outerprodcuct(self, vector_a, vector_b):
        """Keep the legacy misspelled outer-product public helper available."""
        return self.outer_product(vector_a, vector_b)

    def outer_product(self, vector_a, vector_b):
        """Build the matrix of pairwise products between two vectors."""
        return [
            [value_a * value_b for value_b in vector_b]
            for value_a in vector_a
        ]
