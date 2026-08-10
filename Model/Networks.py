"""Handwritten vector LSTM, linear, and temporal projection layers."""

from Model.Node import Node
import math
import random

class Base:
    """Represents shared behavior for layers built from Node objects.

    The base class stores dimensions, learning settings, gradient clipping, and
    the node collection used by concrete LSTM and linear layers.

    Attributes:
        input_size: Number of values accepted by the layer.
        output_size: Number of values produced by the layer.
        learning_rate: Step size used during parameter updates.
        gradient_clip_value: Optional maximum absolute gradient value.
        nodes: Trainable Node objects owned by the layer.
    """

    def __init__(self, input_size, learning_rate, output_size=None):
        """
        Stores the common network configuration and creates its nodes.
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.gradient_clip_value = None
        self.output_size = input_size if output_size is None else output_size
        self.nodes = []
        self.set_up()

    def set_up(self):
        """
        Creates the node objects used by the network.
        """
        raise NotImplementedError("subclasses must implement set_up")

    def run(self, input):
        """
        Sends one input vector through every node in the network.
        """
        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")

        for i in range(len(self.nodes)):
            self.nodes[i].node_run(input)

        return [node.output for node in self.nodes]

    def backpropegate(self, new_weights, new_biases):
        """
        Passes weight and bias updates to each node in the network.
        """
        for i in range(len(self.nodes)):
            self.nodes[i].backpropegate(new_weights[i], new_biases[i])

    def limit_gradient(self, gradient):
        """
        Rejects non-finite gradients and optionally clips large updates.
        """
        if not math.isfinite(gradient):
            raise ValueError("gradient must be finite")
        if self.gradient_clip_value is None:
            return gradient
        return max(-self.gradient_clip_value, min(self.gradient_clip_value, gradient))

    def updated_parameter(self, parameter, gradient):
        """Return one finite parameter update after optional gradient clipping."""
        updated_value = parameter - (
            self.learning_rate * self.limit_gradient(gradient)
        )
        if not math.isfinite(updated_value):
            raise ValueError("updated parameter must be finite")
        return updated_value

    def initialize_xavier_weights(self, fan_in):
        """Initializes every parameter row with Xavier uniform weights."""
        if fan_in <= 0 or self.output_size <= 0:
            raise ValueError("Xavier initialization dimensions must be positive")

        limit = math.sqrt(6 / (fan_in + self.output_size))
        for node in self.nodes:
            for cell in node.cells:
                cell.weights = [
                    random.uniform(-limit, limit)
                    for _ in range(len(cell.weights))
                ]

class Lstm(Base):
    """Represents a vector LSTM layer in the ML library.

    Each gate receives the current input vector and the entire previous hidden
    vector. The layer caches each timestep so backpropagation through time can
    accumulate shared gate gradients before applying an update.

    Attributes:
        input_size: Number of input features per timestep.
        output_size: Number of hidden-state values produced per timestep.
        hidden_state: Most recent hidden-state vector.
        cell_state: Most recent cell-state vector.
        timestep_caches: Forward-pass values used during sequence gradients.
        nodes: One Node per hidden-state dimension.
    """

    def __init__(self, input_size, learning_rate, output_size=None):
        """
        Stores the LSTM configuration and creates its nodes.
        """
        super().__init__(input_size, learning_rate, output_size)
        self.combined_inputs = []
        self.timestep_caches = []

    def set_up(self):
        """
        Creates one parameter row per hidden output for every LSTM gate.
        """
        for i in range(self.output_size):
            self.nodes.append(
                Node(
                    i,
                    self.input_size + self.output_size - 1,
                    self.learning_rate,
                    LSTM=True,
                )
            )
        self.initialize_xavier_weights(self.input_size + self.output_size)

    def run(self, input):
        """
        Sends one input vector through a vector LSTM and caches the timestep.
        """
        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")

        previous_hidden = getattr(self, "hidden_state", [0 for _ in self.nodes])
        previous_cellstate = getattr(self, "cell_state", [0 for _ in self.nodes])
        # Every gate sees the current features and the full previous hidden
        # vector, allowing recurrent connections between hidden dimensions.
        combined_input = input[:] + previous_hidden[:]
        expected_size = self.input_size + self.output_size
        if len(combined_input) != expected_size:
            raise ValueError("combined input must contain input and hidden-state values")

        gate_outputs = []
        for gate_index in range(4):
            outputs = []
            for node in self.nodes:
                cell = node.cells[gate_index]
                cell.cell_run(combined_input)
                outputs.append(cell.output)
            gate_outputs.append(outputs)

        forget_gate, input_gate, output_gate, candidate = gate_outputs
        cell_state = [
            forget * previous_cell + input_value * candidate_value
            for forget, previous_cell, input_value, candidate_value in zip(
                forget_gate,
                previous_cellstate,
                input_gate,
                candidate,
            )
        ]
        tanh_cellstate = [math.tanh(value) for value in cell_state]
        hidden_state = [
            output * tanh_value
            for output, tanh_value in zip(output_gate, tanh_cellstate)
        ]

        for node_index, node in enumerate(self.nodes):
            node.previous_cellstate = previous_cellstate[node_index]
            node.combined_input = combined_input[:]
            node.forget_gate_output = forget_gate[node_index]
            node.input_gate_output = input_gate[node_index]
            node.output_gate_output = output_gate[node_index]
            node.candidate_output = candidate[node_index]
            node.cellstate = cell_state[node_index]
            node.tanh_cellstate = tanh_cellstate[node_index]
            node.output = hidden_state[node_index]

        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.combined_inputs = [combined_input[:] for _ in self.nodes]
        # Backpropagation through time uses these immutable forward values.
        self.timestep_caches.append(
            {
                "combined_input": combined_input[:],
                "previous_hidden": previous_hidden[:],
                "previous_cellstate": previous_cellstate[:],
                "forget_gate_output": forget_gate[:],
                "input_gate_output": input_gate[:],
                "output_gate_output": output_gate[:],
                "candidate_output": candidate[:],
                "cellstate": cell_state[:],
                "tanh_cellstate": tanh_cellstate[:],
            }
        )

        return hidden_state[:]

    def reset_state(self):
        """
        Clears recurrent state and per-run caches before an independent input.
        """
        cached_node_values = [
            "previous_cellstate",
            "combined_input",
            "forget_gate_output",
            "input_gate_output",
            "output_gate_output",
            "candidate_output",
            "tanh_cellstate",
        ]
        cached_lstm_values = [
            "dL_dz_forget",
            "dL_dz_input",
            "dL_dz_output",
            "dL_dz_candidate",
            "dL_dinput",
            "ddecoded_dFinal_weights",
            "ddecoded_dFinal_biases",
            "new_weights",
            "new_biases",
        ]

        for node in self.nodes:
            node.output = None
            node.cellstate = None
            for attribute in cached_node_values:
                if hasattr(node, attribute):
                    delattr(node, attribute)

        self.combined_inputs = []
        self.timestep_caches = []
        self.hidden_state = [0 for _ in self.nodes]
        self.cell_state = [0 for _ in self.nodes]
        for attribute in cached_lstm_values:
            if hasattr(self, attribute):
                delattr(self, attribute)

    def backpropegate(self, dL_doutput):
        """
        Calculates LSTM gate gradients and returns the input gradient.
        """
        dL_dinputs = self.backpropegate_sequence([dL_doutput])
        self.dL_dinput = dL_dinputs[0]
        return self.dL_dinput

    def backpropegate_sequence(self, dL_doutputs):
        """
        Backpropagates through a vector LSTM and updates shared gates once.
        """
        if len(dL_doutputs) != len(self.timestep_caches):
            raise ValueError(
                "sequence gradients must have one vector for each cached LSTM timestep"
            )

        gate_names = ["forget", "input", "output", "candidate"]
        weight_gradients = {
            gate_name: [
                [0 for _ in range(self.input_size + self.output_size)]
                for _ in self.nodes
            ]
            for gate_name in gate_names
        }
        bias_gradients = {
            gate_name: [0 for _ in self.nodes]
            for gate_name in gate_names
        }
        dL_dinputs = [
            [0 for _ in range(self.input_size)]
            for _ in dL_doutputs
        ]
        dL_dnext_hidden = [0 for _ in self.nodes]
        dL_dnext_cellstate = [0 for _ in self.nodes]

        # Traverse the sequence backward, carrying gradients from the next
        # timestep through both the hidden state and the cell state.
        for timestep_index in range(len(dL_doutputs) - 1, -1, -1):
            dL_doutput = dL_doutputs[timestep_index]
            timestep_cache = self.timestep_caches[timestep_index]

            if len(dL_doutput) != len(self.nodes):
                raise ValueError(
                    "each sequence gradient must have one value for each LSTM node"
                )

            combined_input = timestep_cache["combined_input"]
            if len(combined_input) != self.input_size + self.output_size:
                raise ValueError("cached combined_input has the wrong size")

            dL_dhidden = [
                upstream_gradient + next_gradient
                for upstream_gradient, next_gradient in zip(
                    dL_doutput,
                    dL_dnext_hidden,
                )
            ]
            dL_dcellstate = [
                hidden_gradient * output_gate * (1 - tanh_cellstate ** 2)
                + next_cellstate_gradient
                for hidden_gradient, output_gate, tanh_cellstate, next_cellstate_gradient in zip(
                    dL_dhidden,
                    timestep_cache["output_gate_output"],
                    timestep_cache["tanh_cellstate"],
                    dL_dnext_cellstate,
                )
            ]
            dL_dz = {
                "forget": [
                    cell_gradient * previous_cellstate * forget_gate * (1 - forget_gate)
                    for cell_gradient, previous_cellstate, forget_gate in zip(
                        dL_dcellstate,
                        timestep_cache["previous_cellstate"],
                        timestep_cache["forget_gate_output"],
                    )
                ],
                "input": [
                    cell_gradient * candidate * input_gate * (1 - input_gate)
                    for cell_gradient, candidate, input_gate in zip(
                        dL_dcellstate,
                        timestep_cache["candidate_output"],
                        timestep_cache["input_gate_output"],
                    )
                ],
                "output": [
                    hidden_gradient * tanh_cellstate * output_gate * (1 - output_gate)
                    for hidden_gradient, tanh_cellstate, output_gate in zip(
                        dL_dhidden,
                        timestep_cache["tanh_cellstate"],
                        timestep_cache["output_gate_output"],
                    )
                ],
                "candidate": [
                    cell_gradient * input_gate * (1 - candidate ** 2)
                    for cell_gradient, input_gate, candidate in zip(
                        dL_dcellstate,
                        timestep_cache["input_gate_output"],
                        timestep_cache["candidate_output"],
                    )
                ],
            }
            dL_dcombined_input = [0 for _ in combined_input]

            for gate_index, gate_name in enumerate(gate_names):
                for node_index, gate_gradient in enumerate(dL_dz[gate_name]):
                    gate_weights = self.nodes[node_index].cells[gate_index].weights
                    for input_index, input_value in enumerate(combined_input):
                        weight_gradients[gate_name][node_index][input_index] += (
                            gate_gradient * input_value
                        )
                        dL_dcombined_input[input_index] += (
                            gate_gradient * gate_weights[input_index]
                        )
                    bias_gradients[gate_name][node_index] += gate_gradient

            dL_dinputs[timestep_index] = dL_dcombined_input[:self.input_size]
            dL_dnext_hidden = dL_dcombined_input[self.input_size:]
            dL_dnext_cellstate = [
                cell_gradient * forget_gate
                for cell_gradient, forget_gate in zip(
                    dL_dcellstate,
                    timestep_cache["forget_gate_output"],
                )
            ]

        self.ddecoded_dFinal_weights = weight_gradients
        self.ddecoded_dFinal_biases = bias_gradients
        self.validate_all_gate_gradients(weight_gradients, bias_gradients)
        # Gate parameters are shared across timesteps, so apply one update
        # only after their full sequence gradients have been accumulated.
        self.set_weights()
        self.set_bias()

        return dL_dinputs

    def set_weights(self):
        """Build updated LSTM gate weight rows from cached sequence gradients."""
        gate_names = ["forget", "input", "output", "candidate"]
        new_weights = []

        for node_index, node in enumerate(self.nodes):
            node_weights = []

            for gate_index, gate_name in enumerate(gate_names):
                old_weights = node.cells[gate_index].weights
                weight_gradients = self.ddecoded_dFinal_weights[gate_name][node_index]

                if len(old_weights) != len(weight_gradients):
                    raise ValueError(
                        f"{gate_name} weight gradients must match the old weights"
                    )

                node_weights.append([
                    self.updated_parameter(old_weight, dloss_dweight)
                    for old_weight, dloss_dweight in zip(old_weights, weight_gradients)
                ])

            new_weights.append(node_weights)

        self.new_weights = new_weights
        return new_weights

    def set_bias(self):
        """Build and apply updated LSTM gate biases for every hidden node."""
        gate_names = ["forget", "input", "output", "candidate"]
        new_biases = []

        for node_index, node in enumerate(self.nodes):
            node_biases = []

            for gate_index, gate_name in enumerate(gate_names):
                old_bias = node.cells[gate_index].bias
                dloss_dbias = self.ddecoded_dFinal_biases[gate_name][node_index]
                node_biases.append(self.updated_parameter(old_bias, dloss_dbias))

            new_biases.append(node_biases)

        self.new_biases = new_biases

        for i in range(len(self.nodes)):
            self.nodes[i].backpropegate(
                self.new_weights[i],
                self.new_biases[i],
            )

        return new_biases

    def weight_calculation(self, dL_doutput):
        """Calculate legacy one-timestep gate weight gradients."""
        if len(dL_doutput) != len(self.nodes):
            raise ValueError("dL_doutput must have one value for each LSTM node")

        dL_dz_forget = []
        dL_dz_input = []
        dL_dz_output = []
        dL_dz_candidate = []
        dL_dcombined_input = [0 for _ in range(self.input_size + 1)]

        for upstream_gradient, node in zip(dL_doutput, self.nodes):
            if not hasattr(node, "combined_input"):
                raise ValueError("Lstm.run must be called before Lstm.backpropegate")
            if len(node.combined_input) != self.input_size + 1:
                raise ValueError("cached combined_input has the wrong size")

            dL_dcellstate = (
                upstream_gradient
                * node.output_gate_output
                * (1 - (node.tanh_cellstate ** 2))
            )

            dL_dforget = dL_dcellstate * node.previous_cellstate
            dL_dinput = dL_dcellstate * node.candidate_output
            dL_doutput_gate = upstream_gradient * node.tanh_cellstate
            dL_dcandidate = dL_dcellstate * node.input_gate_output

            node_dL_dz_forget = (
                dL_dforget
                * node.forget_gate_output
                * (1 - node.forget_gate_output)
            )
            node_dL_dz_input = (
                dL_dinput
                * node.input_gate_output
                * (1 - node.input_gate_output)
            )
            node_dL_dz_output = (
                dL_doutput_gate
                * node.output_gate_output
                * (1 - node.output_gate_output)
            )
            node_dL_dz_candidate = (
                dL_dcandidate
                * (1 - (node.candidate_output ** 2))
            )

            dL_dz_forget.append(node_dL_dz_forget)
            dL_dz_input.append(node_dL_dz_input)
            dL_dz_output.append(node_dL_dz_output)
            dL_dz_candidate.append(node_dL_dz_candidate)

            gate_gradients = [
                (node_dL_dz_forget, node.cells[0].weights),
                (node_dL_dz_input, node.cells[1].weights),
                (node_dL_dz_output, node.cells[2].weights),
                (node_dL_dz_candidate, node.cells[3].weights),
            ]

            for gate_gradient, gate_weights in gate_gradients:
                for i in range(len(gate_weights)):
                    dL_dcombined_input[i] += gate_gradient * gate_weights[i]

        self.dL_dz_forget = dL_dz_forget
        self.dL_dz_input = dL_dz_input
        self.dL_dz_output = dL_dz_output
        self.dL_dz_candidate = dL_dz_candidate
        self.dL_dinput = dL_dcombined_input[:self.input_size]

        ddecoded_dforget_weights = self.outer_product(dL_dz_forget, self.combined_inputs)
        ddecoded_dinput_weights = self.outer_product(dL_dz_input, self.combined_inputs)
        ddecoded_doutput_weights = self.outer_product(dL_dz_output, self.combined_inputs)
        ddecoded_dcandidate_weights = self.outer_product(dL_dz_candidate, self.combined_inputs)

        ddecoded_dFinal_weights = {
            "forget": ddecoded_dforget_weights,
            "input": ddecoded_dinput_weights,
            "output": ddecoded_doutput_weights,
            "candidate": ddecoded_dcandidate_weights,
        }

        self.ddecoded_dFinal_weights = ddecoded_dFinal_weights
        return ddecoded_dFinal_weights

    def bias_caluclation(self):
        """Return legacy one-timestep gate bias gradients."""
        ddecoded_dforget_biases = self.dL_dz_forget
        ddecoded_dinput_biases = self.dL_dz_input
        ddecoded_doutput_biases = self.dL_dz_output
        ddecoded_dcandidate_biases = self.dL_dz_candidate

        ddecoded_dFinal_biases = {
            "forget": ddecoded_dforget_biases,
            "input": ddecoded_dinput_biases,
            "output": ddecoded_doutput_biases,
            "candidate": ddecoded_dcandidate_biases,
        }

        self.ddecoded_dFinal_biases = ddecoded_dFinal_biases
        self.validate_all_gate_gradients(
            self.ddecoded_dFinal_weights,
            self.ddecoded_dFinal_biases,
        )

        return ddecoded_dFinal_biases

    def validate_all_gate_gradients(self, weight_gradients, bias_gradients):
        """Validate every gate's gradient dimensions before an update."""
        gate_names = ["forget", "input", "output", "candidate"]

        for gate_index, gate_name in enumerate(gate_names):
            weights = [node.cells[gate_index].weights for node in self.nodes]
            biases = [node.cells[gate_index].bias for node in self.nodes]
            self.validate_gate_gradients(
                gate_name,
                weight_gradients[gate_name],
                bias_gradients[gate_name],
                weights,
                biases,
            )

    def outer_product(self, vector_a, vector_b):
        """
        Creates one gradient row per gate output using its cached input row.
        """
        if len(vector_a) != len(vector_b):
            raise ValueError("outer_product inputs must have the same number of rows")

        return [
            [value_a * value_b for value_b in row_b]
            for value_a, row_b in zip(vector_a, vector_b)
        ]

    def validate_gate_gradients(self, gate_name, weight_gradient, bias_gradient, weights, biases):
        """
        Validates a gate's gradients before applying parameter updates.
        """
        if len(weight_gradient) != len(weights):
            raise ValueError(
                f"{gate_name} weight gradient must have {len(weights)} rows"
            )

        for i in range(len(weights)):
            if len(weight_gradient[i]) != len(weights[i]):
                raise ValueError(
                    f"{gate_name} weight gradient row {i} must have "
                    f"{len(weights[i])} values"
                )

        if len(bias_gradient) != len(biases):
            raise ValueError(
                f"{gate_name} bias gradient must have {len(biases)} values"
            )


class Linear(Base):
    """Represents a feature projection built from non-LSTM nodes.

    The layer transforms each input vector independently and caches inputs for
    sequence-level gradient accumulation. It shrinks or expands latent feature
    width without changing the number of timesteps.

    Attributes:
        input_size: Number of input features per vector.
        output_size: Number of projected features per vector.
        last_input: Most recent input used by single-vector backpropagation.
        timestep_inputs: Cached inputs used by sequence backpropagation.
        nodes: One Node per output feature.
    """

    def __init__(self, input_size, learning_rate, bottlenck):
        """Create a feature projection with the requested output width."""
        self.bottlenck = bottlenck
        super().__init__(input_size, learning_rate, bottlenck)
        self.last_input = None
        self.timestep_inputs = []

    def set_up(self):
        """
        Creates the node objects used by this linear network.
        """
        for i in range(self.output_size):
            self.nodes.append(Node(i, self.input_size, self.learning_rate, LSTM=False))
        self.initialize_xavier_weights(self.input_size + 1)

    def run(self, input):
        """
        Sends one input vector through every linear node and caches it.
        """
        self.last_input = input[:]
        self.timestep_inputs.append(input[:])
        return super().run(input)

    def reset_cache(self):
        """
        Clears cached inputs before a new independent sequence.
        """
        self.last_input = None
        self.timestep_inputs = []

    def backpropegate(self, dL_doutput):
        """
        Updates this linear layer and returns the gradient for its input.
        """
        if self.last_input is None:
            raise ValueError("Linear.run must be called before Linear.backpropegate")
        if len(dL_doutput) != len(self.nodes):
            raise ValueError("dL_doutput must have one value for each linear output node")
        if len(self.last_input) != self.input_size:
            raise ValueError("cached linear input has the wrong size")

        combined_input = [0] + self.last_input
        dL_dinput = [0 for _ in range(self.input_size)]

        for upstream_gradient, node in zip(dL_doutput, self.nodes):
            cell = node.cells[0]

            if len(combined_input) != len(cell.weights):
                raise ValueError("linear weight gradient must match the cell weight size")

            weight_gradient = [
                self.limit_gradient(upstream_gradient * value)
                for value in combined_input
            ]
            bias_gradient = self.limit_gradient(upstream_gradient)

            for i in range(self.input_size):
                dL_dinput[i] += upstream_gradient * cell.weights[i + 1]

            node.backpropegate(weight_gradient, bias_gradient)

        return dL_dinput

    def backpropegate_sequence(self, dL_doutputs):
        """
        Accumulates gradients from every cached linear timestep before updating.
        """
        if len(dL_doutputs) != len(self.timestep_inputs):
            raise ValueError(
                "sequence gradients must have one vector for each cached linear timestep"
            )

        weight_gradients = [
            [0 for _ in range(self.input_size + 1)]
            for _ in self.nodes
        ]
        bias_gradients = [0 for _ in self.nodes]
        dL_dinputs = [
            [0 for _ in range(self.input_size)]
            for _ in dL_doutputs
        ]

        for timestep_index, (dL_doutput, cached_input) in enumerate(
            zip(dL_doutputs, self.timestep_inputs)
        ):
            if len(dL_doutput) != len(self.nodes):
                raise ValueError(
                    "each sequence gradient must have one value for each linear output node"
                )

            combined_input = [0] + cached_input
            for node_index, (upstream_gradient, node) in enumerate(
                zip(dL_doutput, self.nodes)
            ):
                cell = node.cells[0]
                if len(combined_input) != len(cell.weights):
                    raise ValueError(
                        "linear weight gradient must match the cell weight size"
                    )

                for input_index, input_value in enumerate(combined_input):
                    weight_gradients[node_index][input_index] += (
                        upstream_gradient * input_value
                    )
                bias_gradients[node_index] += upstream_gradient

                for input_index in range(self.input_size):
                    dL_dinputs[timestep_index][input_index] += (
                        upstream_gradient * cell.weights[input_index + 1]
                    )

        for node, weight_gradient, bias_gradient in zip(
            self.nodes,
            weight_gradients,
            bias_gradients,
        ):
            node.backpropegate(
                [self.limit_gradient(gradient) for gradient in weight_gradient],
                self.limit_gradient(bias_gradient),
            )

        return dL_dinputs


class TemporalProjection:
    """Represents a learnable projection between two sequence lengths.

    The projection applies one time-weight matrix to every feature column, so
    it changes the number of timesteps without mixing feature dimensions. It is
    used for temporal compression and expansion in the autoencoder.

    Attributes:
        input_steps: Number of timesteps accepted by the projection.
        output_steps: Number of timesteps produced by the projection.
        feature_size: Number of unchanged features at every timestep.
        weights: Trainable time-to-time projection matrix.
        last_input: Cached sequence used to calculate weight gradients.
    """

    def __init__(self, input_steps, output_steps, feature_size, learning_rate):
        """Create a learnable mapping between two sequence lengths."""
        if input_steps <= 0 or output_steps <= 0 or feature_size <= 0:
            raise ValueError("temporal projection dimensions must be greater than zero")

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.feature_size = feature_size
        self.learning_rate = learning_rate
        self.gradient_clip_value = None
        self.weights = self.initial_weights()
        self.last_input = None
        self.dL_dweights = None

    def initial_weights(self):
        """Initialize averaging or repeat-style temporal projection weights."""
        weights = []
        for output_index in range(self.output_steps):
            row = [0 for _ in range(self.input_steps)]
            if self.output_steps <= self.input_steps:
                start_index = (output_index * self.input_steps) // self.output_steps
                end_index = ((output_index + 1) * self.input_steps) // self.output_steps
                span = end_index - start_index
                for input_index in range(start_index, end_index):
                    row[input_index] = 1 / span
            else:
                # Initial temporal upsampling repeats the nearest latent step.
                input_index = (output_index * self.input_steps) // self.output_steps
                row[input_index] = 1
            weights.append(row)
        return weights

    def run(self, sequence):
        """Project a sequence across time while retaining feature columns."""
        self.validate_sequence(sequence, self.input_steps)
        self.last_input = [row[:] for row in sequence]

        return [
            [
                sum(
                    self.weights[output_index][input_index]
                    * sequence[input_index][feature_index]
                    for input_index in range(self.input_steps)
                )
                for feature_index in range(self.feature_size)
            ]
            for output_index in range(self.output_steps)
        ]

    def backpropegate(self, dL_doutput):
        """Update temporal weights and return gradients for input timesteps."""
        if self.last_input is None:
            raise ValueError("TemporalProjection.run must be called before backpropegate")
        self.validate_sequence(dL_doutput, self.output_steps)

        dL_dinput = [
            [0 for _ in range(self.feature_size)]
            for _ in range(self.input_steps)
        ]
        dL_dweights = [
            [0 for _ in range(self.input_steps)]
            for _ in range(self.output_steps)
        ]

        for output_index in range(self.output_steps):
            for input_index in range(self.input_steps):
                for feature_index in range(self.feature_size):
                    gradient = dL_doutput[output_index][feature_index]
                    dL_dinput[input_index][feature_index] += (
                        gradient * self.weights[output_index][input_index]
                    )
                    dL_dweights[output_index][input_index] += (
                        gradient * self.last_input[input_index][feature_index]
                    )

        self.dL_dweights = dL_dweights
        for output_index in range(self.output_steps):
            for input_index in range(self.input_steps):
                gradient = self.limit_gradient(
                    dL_dweights[output_index][input_index]
                )
                self.weights[output_index][input_index] -= (
                    self.learning_rate * gradient
                )

        return dL_dinput

    def limit_gradient(self, gradient):
        """Reject non-finite temporal gradients and clip configured extremes."""
        if not math.isfinite(gradient):
            raise ValueError("temporal projection gradient must be finite")
        if self.gradient_clip_value is None:
            return gradient
        return max(-self.gradient_clip_value, min(self.gradient_clip_value, gradient))

    def validate_sequence(self, sequence, expected_steps):
        """Check temporal length and feature width before a projection."""
        if len(sequence) != expected_steps:
            raise ValueError(
                f"temporal sequence must contain {expected_steps} timesteps"
            )
        for timestep in sequence:
            if len(timestep) != self.feature_size:
                raise ValueError(
                    f"temporal sequence timesteps must contain {self.feature_size} features"
                )
