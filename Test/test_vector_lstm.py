"""Vector-LSTM recurrence and numerical-gradient tests."""

import unittest

from Model.Networks import Lstm


class VectorLstmTests(unittest.TestCase):
    def test_gate_rows_include_the_full_previous_hidden_vector(self):
        network = Lstm(input_size=1, learning_rate=0.001, output_size=2)

        for node in network.nodes:
            for cell in node.cells:
                self.assertEqual(len(cell.weights), 3)

    def test_cross_hidden_weight_gradient_matches_finite_difference(self):
        sequence = [[0.2], [-0.1], [0.3]]
        targets = [[0.1, -0.05], [-0.2, 0.1], [0.15, 0.2]]
        network = Lstm(input_size=1, learning_rate=0.001, output_size=2)

        for node_index, node in enumerate(network.nodes):
            for gate_index, cell in enumerate(node.cells):
                cell.weights = [
                    0.02 * (node_index + 1),
                    -0.03 * (gate_index + 1),
                    0.04 * (node_index + gate_index + 1),
                ]
                cell.bias = 0.01 * (gate_index - 1)

        original_parameters = [
            [(cell.weights[:], cell.bias) for cell in node.cells]
            for node in network.nodes
        ]

        def restore_parameters():
            for node, node_parameters in zip(network.nodes, original_parameters):
                for cell, (weights, bias) in zip(node.cells, node_parameters):
                    cell.weights = weights[:]
                    cell.bias = bias

        def loss():
            network.reset_state()
            outputs = [network.run(timestep) for timestep in sequence]
            return 0.5 * sum(
                (output - target) ** 2
                for output_vector, target_vector in zip(outputs, targets)
                for output, target in zip(output_vector, target_vector)
            )

        restore_parameters()
        network.reset_state()
        outputs = [network.run(timestep) for timestep in sequence]
        network.backpropegate_sequence(
            [
                [output - target for output, target in zip(output_vector, target_vector)]
                for output_vector, target_vector in zip(outputs, targets)
            ]
        )
        analytic_gradient = network.ddecoded_dFinal_weights["candidate"][0][2]

        epsilon = 1e-6
        restore_parameters()
        network.nodes[0].cells[3].weights[2] += epsilon
        positive_loss = loss()
        restore_parameters()
        network.nodes[0].cells[3].weights[2] -= epsilon
        negative_loss = loss()
        numerical_gradient = (positive_loss - negative_loss) / (2 * epsilon)

        self.assertAlmostEqual(analytic_gradient, numerical_gradient, places=9)


if __name__ == "__main__":
    unittest.main()
