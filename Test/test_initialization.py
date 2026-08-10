"""Initialization bounds for trainable layer parameters."""

import math
import random
import unittest

from Model.Networks import Linear, Lstm


class XavierInitializationTests(unittest.TestCase):
    def test_lstm_uses_xavier_bound_for_vector_gate_rows(self):
        random.seed(42)
        network = Lstm(input_size=1, learning_rate=0.001, output_size=8)
        limit = math.sqrt(6 / (9 + 8))

        for node in network.nodes:
            for cell in node.cells:
                self.assertEqual(len(cell.weights), 9)
                self.assertTrue(all(-limit <= weight <= limit for weight in cell.weights))

    def test_linear_uses_xavier_bound(self):
        random.seed(42)
        network = Linear(input_size=8, learning_rate=0.001, bottlenck=4)
        limit = math.sqrt(6 / (9 + 4))

        for node in network.nodes:
            for cell in node.cells:
                self.assertEqual(len(cell.weights), 9)
                self.assertTrue(all(-limit <= weight <= limit for weight in cell.weights))


if __name__ == "__main__":
    unittest.main()
