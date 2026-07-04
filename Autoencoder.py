from Networks import Lstm, Linear
from sympy import symbols, diff

class Autoencoder:
    
    def __init__(self, input_size, learning_rate, bottleneck, loss_type="mse", target_type="early_stopping", target=10, bidirectional=False):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.bottleneck = bottleneck
        self.bidirectional = bidirectional
        self.target_type = target_type
        self.target = target
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

            loss_vector = self.compute_loss(decoded, input)
            loss_mean = sum(loss_vector) / len(loss_vector)
            validation.append(loss_mean)

            if  not self.check_stop(validation, target):
                self.backpropegate(loss_vector, input, decoded)

            else:
                stopper = True

        return decoded
    
    def check_stop(self, rounds):
        if self.target_type == "early_stopping":
            if len(rounds) >= self.target:
                for i in range(len(rounds) - self.target, len(rounds) - 1):
                    if rounds[i] < rounds[i + 1]:
                        return False
            else:
                return False
        else:
            return True
        
    def compute_loss(self, decoded, input):
        loss_vector = list()
        for i in range(len(decoded)):
            if self.loss_type == "mse":
                loss_vector.append((decoded[i] - input[i]) ** 2)
        return loss_vector

    def backpropegate(self, loss_vector, input, decoded):
        if self.loss_type == "mse":
            loss_vector_weights = [self.pd_loss_pd_y(actual, predicted) for actual, predicted in zip(input, decoded)]
        else:
            raise NotImplementedError(
                f"Loss type '{self.loss_type}' has not been implemented yet.")
        #new_weights_encoder = [weight - self.learning_rate * gradient for weight, gradient in zip(self.encoder.weights, loss_vector_weights)]
        self.encoder.backpropegate(new_weights_encoder, new_biases_encoder)
        self.linear_encoder.backpropegate(new_weights_linear_encoder, new_biases_linear_encoder)
        self.linear_decoder.backpropegate(new_weights_linear_decoder, new_biases_linear_decoder)
        self.decoder.backpropegate(new_weights_decoder, new_biases_decoder)

    def pd_loss_pd_y(self, actual, predicted):
        return 2 * (predicted - actual)
