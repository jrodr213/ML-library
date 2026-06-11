from Networks import Lstm

class Autoencoder:
    
    def __init__(self, input_size, learning_rate, Bidirectional=False):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.bidirectional = Bidirectional
        self.encoder = Lstm(input_size, learning_rate)
        self.decoder = Lstm(input_size, learning_rate)

    def run(self, input):
        if len(input) != self.input_size:
            raise ValueError("input must be the same size as the input size configuration")
        
        encoded = self.encoder.run(input)
        decoded = self.decoder.run(encoded)
        return decoded

    def backpropegate(self, new_weights_encoder, new_biases_encoder, new_weights_decoder, new_biases_decoder):
        self.encoder.backpropegate(new_weights_encoder, new_biases_encoder)
        self.decoder.backpropegate(new_weights_decoder, new_biases_decoder)