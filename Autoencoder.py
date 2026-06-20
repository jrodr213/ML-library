from Networks import Lstm, Linear
from sympy import symbols, diff

class Autoencoder:
    
    def __init__(self, input_size, learning_rate, bottleneck, loss_type="mse", target_type="early_stopping", bidirectional=False):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.bottleneck = bottleneck
        self.bidirectional = bidirectional
        self.target_type = target_type
        self.loss_type = loss_type
        self.encoder = Lstm(input_size, learning_rate)
        self.linear = Linear(input_size, learning_rate, bottleneck)
        self.decoder = Lstm(bottleneck, learning_rate, input_size)

    def run(self, input, target):
        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")
        
        validation = list()

        stopper = False

        while (not stopper):
            encoded = self.encoder.run(input)
            bottleneck_output = self.linear.run(encoded)
            decoded = self.decoder.run(bottleneck_output)

            loss = self.compute_loss(decoded, input)

            if self.target_type == "early_stopping":
                loss_mean = sum(loss) / len(loss)
                validation.append(loss_mean)
                if len(validation) >= target:
                    stopper = self.epoch_check(validation, target)
                else:
                    self.backpropegate(loss)

        return decoded
    
    def epoch_check(self, validation, target):
        if self.targer_type == "early_stopping":
            pass
        else:
            return True
        
    def compute_loss(self, decoded, input):
        loss = list()
        for i in range(len(decoded)):
            if self.loss_type == "mse":
                loss.append((decoded[i] - input[i]) ** 2)
        return loss

    def backpropegate(self, loss):
        self.calculate_pd(loss)
        self.encoder.backpropegate(new_weights_encoder, new_biases_encoder)
        self.linear.backpropegate(new_weights_linear, new_biases_linear)
        self.decoder.backpropegate(new_weights_decoder, new_biases_decoder)

    def calculate_pd(self, matrix, equation):
        pass
