from test.preparations import generate_dataset, get_splitting_params

import unittest

import pandas as pd
from src.data.make_dataset import read_data, split_train_val_data


class TestMakeDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._data_path = 'data/test/test.csv'
        cls._df = generate_dataset(cls._data_path)

    def test_read_data(self):
        actual = read_data(self._data_path)
        pd.testing.assert_frame_equal(actual, self._df)

    def test_split_train_val_data(self):
        val_size = 0.25
        splitting_params = get_splitting_params(val_size=val_size)
        actual_train_df, actual_val_df = split_train_val_data(self._df, splitting_params)
        self.assertEqual(self._df.shape[0], actual_train_df.shape[0] + actual_val_df.shape[0])
        self.assertEqual(self._df.shape[1], actual_train_df.shape[1])
        self.assertEqual(self._df.shape[1], actual_train_df.shape[1])
        self.assertEqual(actual_val_df.shape[0] / self._df.shape[0], val_size)
        self.assertEqual(actual_train_df.shape[0] / self._df.shape[0], 1 - val_size)
