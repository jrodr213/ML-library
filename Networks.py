from Node import Node

class Base:
    """
    Shared parent class for network types built from Node objects.
    """

    def __init__(self, input_size, learning_rate, output_size=None):
        """
        Stores the common network configuration and creates its nodes.
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
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

class Lstm(Base):
    """
    Represents the top-level LSTM structure in the ML library.

    The class stores the shared input size and learning rate, creates a
    collection of nodes, runs input through each node, and forwards
    backpropagation updates down to those nodes.
    """
    
    def __init__(self, input_size, learning_rate, output_size=None):
        """
        Stores the LSTM configuration and creates its nodes.
        """
        super().__init__(input_size, learning_rate, output_size)
        self.combined_inputs = []

    def set_up(self):
        """
        Creates the node objects used by this LSTM.
        """
        for i in range(self.output_size):
            self.nodes.append(Node(i, self.input_size, self.learning_rate, LSTM=True))

    def run(self, input):
        """
        Sends one input vector through every LSTM node and caches gate inputs.
        """
        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")

        self.combined_inputs = []

        for node in self.nodes:
            previous_hidden_state = [0 if node.output is None else node.output]
            previous_cellstate = 0 if node.cellstate is None else node.cellstate
            combined_input = input + previous_hidden_state

            node.previous_cellstate = previous_cellstate
            node.combined_input = combined_input

            for cell in node.cells:
                cell.cell_run(combined_input)

            node.forget_gate_output = node.cells[0].output
            node.input_gate_output = node.cells[1].output
            node.output_gate_output = node.cells[2].output
            node.candidate_output = node.cells[3].output

            node.cellstate = (
                (previous_cellstate * node.forget_gate_output)
                + (node.input_gate_output * node.candidate_output)
            )
            node.tanh_cellstate = node.cells[2].tanh(node.cellstate)
            node.output = node.tanh_cellstate * node.output_gate_output

            self.combined_inputs.append(combined_input)

        return [node.output for node in self.nodes]

    def backpropegate(self, dL_doutput):
        """
        Updates the forget, input, and output gate parameters.
        """
        if len(dL_doutput) != len(self.nodes):
            raise ValueError("dL_doutput must have one value for each LSTM node")

        dL_dz_forget = []
        dL_dz_input = []
        dL_dz_output = []

        for upstream_gradient, node in zip(dL_doutput, self.nodes):
            if not hasattr(node, "combined_input"):
                raise ValueError("Lstm.run must be called before Lstm.backpropegate")

            dL_dcellstate = (
                upstream_gradient
                * node.output_gate_output
                * (1 - (node.tanh_cellstate ** 2))
            )

            dL_dforget = dL_dcellstate * node.previous_cellstate
            dL_dinput = dL_dcellstate * node.candidate_output
            dL_doutput_gate = upstream_gradient * node.tanh_cellstate

            dL_dz_forget.append(
                dL_dforget
                * node.forget_gate_output
                * (1 - node.forget_gate_output)
            )
            dL_dz_input.append(
                dL_dinput
                * node.input_gate_output
                * (1 - node.input_gate_output)
            )
            dL_dz_output.append(
                dL_doutput_gate
                * node.output_gate_output
                * (1 - node.output_gate_output)
            )

        ddecoded_dFinal_weights = self.weight_propegation(
            dL_dz_forget,
            dL_dz_input,
            dL_dz_output,
        )
        ddecoded_dFinal_biases = self.bias_propegation(
            dL_dz_forget,
            dL_dz_input,
            dL_dz_output,
        )

        forget_weights = [node.cells[0].weights for node in self.nodes]
        input_weights = [node.cells[1].weights for node in self.nodes]
        output_weights = [node.cells[2].weights for node in self.nodes]

        forget_biases = [node.cells[0].bias for node in self.nodes]
        input_biases = [node.cells[1].bias for node in self.nodes]
        output_biases = [node.cells[2].bias for node in self.nodes]

        self.validate_gate_gradients(
            "forget",
            ddecoded_dFinal_weights["forget"],
            ddecoded_dFinal_biases["forget"],
            forget_weights,
            forget_biases,
        )
        self.validate_gate_gradients(
            "input",
            ddecoded_dFinal_weights["input"],
            ddecoded_dFinal_biases["input"],
            input_weights,
            input_biases,
        )
        self.validate_gate_gradients(
            "output",
            ddecoded_dFinal_weights["output"],
            ddecoded_dFinal_biases["output"],
            output_weights,
            output_biases,
        )

        return ddecoded_dFinal_weights, ddecoded_dFinal_biases

    def weight_propegation(self, dL_dz_forget, dL_dz_input, dL_dz_output):
        ddecoded_dforget_weights = self.outer_product(dL_dz_forget, self.combined_inputs)
        ddecoded_dinput_weights = self.outer_product(dL_dz_input, self.combined_inputs)
        ddecoded_doutput_weights = self.outer_product(dL_dz_output, self.combined_inputs)

        ddecoded_dFinal_weights = {
            "forget": ddecoded_dforget_weights,
            "input": ddecoded_dinput_weights,
            "output": ddecoded_doutput_weights,
        }

        self.ddecoded_dFinal_weights = ddecoded_dFinal_weights
        return ddecoded_dFinal_weights

    def bias_propegation(self, dL_dz_forget, dL_dz_input, dL_dz_output):
        ddecoded_dforget_biases = dL_dz_forget
        ddecoded_dinput_biases = dL_dz_input
        ddecoded_doutput_biases = dL_dz_output

        ddecoded_dFinal_biases = {
            "forget": ddecoded_dforget_biases,
            "input": ddecoded_dinput_biases,
            "output": ddecoded_doutput_biases,
        }

        self.ddecoded_dFinal_biases = ddecoded_dFinal_biases
        return ddecoded_dFinal_biases

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
    """
    Represents a standard linear-style network built from non-LSTM nodes.
    """
    
    def __init__(self, input_size, learning_rate, bottlenck):
        self.bottlenck = bottlenck
        super().__init__(input_size, learning_rate, bottlenck)

    def set_up(self):
        """
        Creates the node objects used by this linear network.
        """
        for i in range(self.output_size):
            self.nodes.append(Node(i, self.input_size, self.learning_rate, LSTM=False))
