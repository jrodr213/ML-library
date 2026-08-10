"""Temporal compression shape and gradient-flow tests."""

import unittest

import numpy as np

from Model.Autoencoder import Autoencoder


class TemporalBottleneckTests(unittest.TestCase):
    def test_temporal_bottleneck_shapes_and_gradients(self):
        random_batch = np.random.default_rng(42).normal(size=(2, 256, 1)).tolist()
        model = Autoencoder(
            1,
            0.001,
            bottleneck_steps=32,
            hidden_size=8,
            latent_size=4,
            window_size=256,
            max_epochs=1,
        )

        reconstructed_batch = np.asarray(model.reconstruct_batch(random_batch))
        self.assertEqual(reconstructed_batch.shape, (2, 256, 1))
        self.assertEqual(
            model.last_temporal_shapes,
            {
                "input": (256, 1),
                "encoder": (256, 8),
                "compressed": (32, 8),
                "bottleneck": (32, 4),
                "expanded": (256, 8),
                "reconstruction": (256, 1),
            },
        )

        compression_weights = [row[:] for row in model.temporal_compressor.weights]
        expansion_weights = [row[:] for row in model.temporal_expander.weights]
        decoded = model.run(random_batch[0], target=10)

        self.assertEqual(np.asarray(decoded).shape, (256, 1))
        self.assertIsNotNone(model.temporal_compressor.dL_dweights)
        self.assertIsNotNone(model.temporal_expander.dL_dweights)
        self.assertNotEqual(compression_weights, model.temporal_compressor.weights)
        self.assertNotEqual(expansion_weights, model.temporal_expander.weights)

    def test_temporal_bottleneck_trains_corrupted_input_against_clean_target(self):
        clean_sequence = np.ones((8, 1)).tolist()
        corrupted_sequence = np.zeros((8, 1)).tolist()
        model = Autoencoder(
            1,
            0.001,
            bottleneck_steps=2,
            hidden_size=2,
            latent_size=1,
            window_size=8,
            max_epochs=1,
        )

        decoded = model.run(
            corrupted_sequence,
            target=10,
            reconstruction_target=clean_sequence,
        )

        self.assertEqual(np.asarray(decoded).shape, (8, 1))
        self.assertIsNotNone(model.temporal_compressor.dL_dweights)
        self.assertIsNotNone(model.temporal_expander.dL_dweights)


if __name__ == "__main__":
    unittest.main()
