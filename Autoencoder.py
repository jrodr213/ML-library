from Networks import Lstm, Linear
from sympy import symbols, diff

class Autoencoder:
    
    def __init__(self, input_size, learning_rate, bottleneck, loss_type="mse", target_type="early_stopping", target=10, bidirectional=False, max_epochs=None):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.bottleneck = bottleneck
        self.bidirectional = bidirectional
        self.target_type = target_type
        self.target = target
        self.max_epochs = max_epochs
        self.loss_type = loss_type
        self.encoder = Lstm(input_size, learning_rate)
        self.linear_encoder = Linear(input_size, learning_rate, bottleneck)
        self.linear_decoder = Linear(bottleneck, learning_rate, input_size)
        self.decoder = Lstm(bottleneck, learning_rate, input_size)

    def run(self, input, target):
        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")
        
        validation = list()

        stopper = False

        while (not stopper):
            encoded = self.encoder.run(input)
            bottleneck_output = self.linear_encoder.run(encoded)
            bottleneck_input = self.linear_decoder.run(bottleneck_output)
            decoded = self.decoder.run(bottleneck_input)

            loss_list = self.compute_loss(decoded, input)
            loss_mean = sum(loss_list) / len(loss_list)
            validation.append(loss_mean)

            if  not self.check_stop(validation, target):
                self.backpropegate(input, decoded)

            else:
                stopper = True

        return decoded
    
    def check_stop(self, rounds, target=None):
        if self.max_epochs is not None and len(rounds) == self.max_epochs:
            return True

        if target is None:
            target = self.target

        if self.target_type == "early_stopping":
            if len(rounds) >= target:
                for i in range(len(rounds) - target, len(rounds) - 1):
                    if rounds[i] < rounds[i + 1]:
                        return False
            else:
                return False
        else:
            return True
        
    def compute_loss(self, decoded, input):
        loss_list = list()
        for i in range(len(decoded)):
            if self.loss_type == "mse":
                loss_list.append((decoded[i] - input[i]) ** 2)
        return loss_list

    def backpropegate(self, input, decoded):

        self.decoder_propegation(input, decoded)

    def decoder_propegation(self, input, decoded):

        dL_ddecoded = self.pd_loss_pd_y(input, decoded)
        dloss_dweights, dloss_dbiases = self.decoder.backpropegate(dL_ddecoded)
        new_weights_decoder, new_biases_decoder = self.final_lstm_weights_biases(
            self.decoder,
            dloss_dweights,
            dloss_dbiases,
        )

        for i in range(len(self.decoder.nodes)):
            self.decoder.nodes[i].backpropegate(
                new_weights_decoder[i],
                new_biases_decoder[i],
            )


    def pd_loss_pd_y(self, actual, predicted):
        if self.loss_type == "mse":
            return [
                2 * (predicted[i] - actual[i])
                for i in range(len(actual))
            ]
        else:
            raise NotImplementedError(
                f"Loss type '{self.loss_type}' has not been implemented yet.")

    def pd_y_pd_weights(self, input, previous_hidden=None):
        if previous_hidden is None:
            previous_hidden = [0]
        elif not isinstance(previous_hidden, list):
            previous_hidden = [previous_hidden]

        if len(previous_hidden) != 1:
            raise ValueError("previous_hidden must contain exactly one value")

        return previous_hidden + input

    def matrix_outerprodcuct(self, vector_a, vector_b):
        return self.outer_product(vector_a, vector_b)

    def outer_product(self, vector_a, vector_b):
        return [
            [value_a * value_b for value_b in vector_b]
            for value_a in vector_a
        ]

    def final_weights_biases(self, factor, loss_list):

        factor_is_2d = any(isinstance(value, list) for value in factor)
        loss_list_is_2d = any(isinstance(value, list) for value in loss_list)

        if factor_is_2d != loss_list_is_2d:
            raise ValueError("factor and loss_list must both be 1D lists or both be 2D lists")

        if factor_is_2d:
            if len(factor) != len(loss_list):
                raise ValueError("factor and loss_list must have the same number of rows")

            for factor_row, loss_list_row in zip(factor, loss_list):
                if not isinstance(factor_row, list) or not isinstance(loss_list_row, list):
                    raise ValueError("2D factor and loss_list must only contain list rows")
                if len(factor_row) != len(loss_list_row):
                    raise ValueError("factor and loss_list rows must be the same size")

            return [
                [
                    factor_value - (self.learning_rate * loss_value)
                    for factor_value, loss_value in zip(factor_row, loss_list_row)
                ]
                for factor_row, loss_list_row in zip(factor, loss_list)
            ]

        if len(factor) != len(loss_list):
            raise ValueError("factor and loss_list must be the same size")

        return [
            factor_value - (self.learning_rate * loss_value)
            for factor_value, loss_value in zip(factor, loss_list)
        ]

    def final_lstm_weights_biases(self, lstm, dloss_dweights, dloss_dbiases):
        gate_names = ["forget", "input", "output"]
        new_weights = []
        new_biases = []

        for node_index, node in enumerate(lstm.nodes):
            node_weights = []
            node_biases = []

            for gate_index, gate_name in enumerate(gate_names):
                old_weights = node.cells[gate_index].weights
                weight_gradients = dloss_dweights[gate_name][node_index]

                if len(old_weights) != len(weight_gradients):
                    raise ValueError(
                        f"{gate_name} weight gradients must match the old weights"
                    )

                node_weights.append([
                    old_weight - (self.learning_rate * dloss_dweight)
                    for old_weight, dloss_dweight in zip(old_weights, weight_gradients)
                ])

                old_bias = node.cells[gate_index].bias
                dloss_dbias = dloss_dbiases[gate_name][node_index]
                node_biases.append(old_bias - (self.learning_rate * dloss_dbias))

            new_weights.append(node_weights)
            new_biases.append(node_biases)

        return new_weights, new_biases
