"""Construction and backpropagation coverage for directional encoders."""

import unittest

from Model.Autoencoder import Autoencoder
from Model.Networks import Lstm


class BidirectionalAutoencoderTests(unittest.TestCase):
    def test_unidirectional_construction_and_backpropagation(self):
        model = Autoencoder(3, 0.01, 2, bidirectional=False, max_epochs=2)

        self.assertIsInstance(model.encoder, Lstm)
        self.assertEqual(model.encoder.output_size, 3)
        self.assertEqual(model.linear_encoder.input_size, 3)
        self.assertIsInstance(model.decoder, Lstm)
        self.assertEqual(len(model.decoder.nodes), 3)

        decoded = model.run([0.1, 0.2, 0.3], target=10)
        self.assertEqual(len(decoded), 3)

    def test_bidirectional_construction_uses_independent_encoders(self):
        model = Autoencoder(3, 0.01, 2, bidirectional=True, max_epochs=1)

        self.assertIsInstance(model.forward_encoder, Lstm)
        self.assertIsInstance(model.backward_encoder, Lstm)
        self.assertIsNot(
            model.forward_encoder.nodes[0].cells[0].weights,
            model.backward_encoder.nodes[0].cells[0].weights,
        )
        self.assertEqual(model.linear_encoder.input_size, 6)
        self.assertIsInstance(model.decoder, Lstm)
        self.assertEqual(len(model.decoder.nodes), 3)

    def test_bidirectional_output_concatenation_preserves_direction_order(self):
        model = Autoencoder(3, 0.01, 2, bidirectional=True)

        combined = model.combine_encoder_outputs([1, 2, 3], [4, 5, 6])

        self.assertEqual(combined, [1, 2, 3, 4, 5, 6])

    def test_bidirectional_backpropagation_splits_gradient(self):
        model = Autoencoder(3, 0.01, 2, bidirectional=True)
        forward_gradients = []
        backward_gradients = []

        model.forward_encoder.backpropegate = forward_gradients.append
        model.backward_encoder.backpropegate = backward_gradients.append

        model.encoder_propegation([1, 2, 3, 4, 5, 6])

        self.assertEqual(forward_gradients, [[1, 2, 3]])
        self.assertEqual(backward_gradients, [[4, 5, 6]])

    def test_bidirectional_end_to_end_shape(self):
        model = Autoencoder(3, 0.01, 2, bidirectional=True, max_epochs=2)

        decoded = model.run([0.1, 0.2, 0.3], target=10)

        self.assertEqual(len(decoded), 3)

    def test_bidirectional_sequence_reverses_and_realigns_outputs(self):
        model = Autoencoder(3, 0.01, 2, bidirectional=True, max_epochs=1)
        sequence = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
        forward_inputs = []
        backward_inputs = []
        linear_encoder_inputs = []

        forward_run = model.forward_encoder.run
        backward_run = model.backward_encoder.run
        linear_encoder_run = model.linear_encoder.run

        def record_forward(timestep):
            forward_inputs.append(timestep[:])
            forward_run(timestep)
            return [timestep[0]] * 3

        def record_backward(timestep):
            backward_inputs.append(timestep[:])
            backward_run(timestep)
            return [timestep[0] * 10] * 3

        def record_linear_encoder(encoded):
            linear_encoder_inputs.append(encoded[:])
            return linear_encoder_run(encoded)

        model.forward_encoder.run = record_forward
        model.backward_encoder.run = record_backward
        model.linear_encoder.run = record_linear_encoder

        model.run(sequence, target=10)

        self.assertEqual(forward_inputs, sequence)
        self.assertEqual(backward_inputs, list(reversed(sequence)))
        self.assertEqual(
            linear_encoder_inputs,
            [
                [1, 1, 1, 10, 10, 10],
                [2, 2, 2, 20, 20, 20],
                [3, 3, 3, 30, 30, 30],
            ],
        )

    def test_bidirectional_sequence_backpropagation_updates_each_encoder(self):
        model = Autoencoder(3, 0.01, 2, bidirectional=True, max_epochs=2)
        sequence = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        forward_weight = model.forward_encoder.nodes[0].cells[0].weights[:]
        backward_weight = model.backward_encoder.nodes[0].cells[0].weights[:]

        decoded = model.run(sequence, target=10)

        self.assertEqual(len(decoded), 2)
        self.assertTrue(all(len(timestep) == 3 for timestep in decoded))
        self.assertNotEqual(
            forward_weight,
            model.forward_encoder.nodes[0].cells[0].weights,
        )
        self.assertNotEqual(
            backward_weight,
            model.backward_encoder.nodes[0].cells[0].weights,
        )

    def test_one_max_epoch_performs_an_update(self):
        model = Autoencoder(3, 0.01, 2, max_epochs=1)
        candidate_weights = model.encoder.nodes[0].cells[3].weights[:]

        model.run([0.1, 0.2, 0.3], target=10)

        self.assertNotEqual(
            candidate_weights,
            model.encoder.nodes[0].cells[3].weights,
        )

    def test_early_stopping_requires_no_recent_improvement(self):
        model = Autoencoder(3, 0.01, 2, max_epochs=None)

        self.assertFalse(model.check_stop([1.0, 0.5, 0.4], target=2))
        self.assertTrue(model.check_stop([1.0, 0.5, 0.5, 0.5], target=2))


if __name__ == "__main__":
    unittest.main()
