"""Chronological split and sliding-window utility tests."""

import unittest

from Model.Autoencoder import (
    Autoencoder,
    STEP_SIZE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    WINDOW_SIZE,
)


class WindowingTests(unittest.TestCase):
    def test_default_window_size_and_step_size(self):
        rows = [[float(index), float(index + 1)] for index in range(50)]

        windows = Autoencoder.create_sliding_windows(rows, WINDOW_SIZE, STEP_SIZE)

        self.assertEqual(WINDOW_SIZE, 36)
        self.assertEqual(STEP_SIZE, 3)
        self.assertEqual(len(windows), 5)
        self.assertEqual((len(windows[0]), len(windows[0][0])), (36, 2))

    def test_split_happens_before_windows(self):
        rows = [[float(index), float(index + 1)] for index in range(180)]

        training_rows, testing_rows, training_windows, testing_windows = (
            Autoencoder.prepare_windowed_splits(
                rows,
                WINDOW_SIZE,
                STEP_SIZE,
                TRAIN_SPLIT,
                TEST_SPLIT,
            )
        )

        self.assertEqual(len(training_rows), 144)
        self.assertEqual(len(testing_rows), 36)
        self.assertEqual(len(training_windows), 37)
        self.assertEqual(len(testing_windows), 1)
        self.assertEqual(training_windows[-1][-1][0], 143.0)
        self.assertEqual(testing_windows[0][0][0], 144.0)


if __name__ == "__main__":
    unittest.main()
