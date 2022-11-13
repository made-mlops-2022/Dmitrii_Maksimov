from test.preparations import generate_dataset, get_feature_params

import unittest

import pandas as pd
import numpy as np
from src.features.build_features import build_transformer, extract_target


class TestBuildFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._data_path = 'data/test/test.csv'
        cls._df = generate_dataset(cls._data_path)
        cls._categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
        cls._numerical_features = ["trestbps", "chol", "thalach", "oldpeak", "ca"]
        cls._features_to_drop = ["age"]
        cls._target_col = "condition"
        cls._feature_params = get_feature_params(
            categorical_features=cls._categorical_features,
            numerical_features=cls._numerical_features,
            features_to_drop=cls._features_to_drop,
            target_col=cls._target_col,
            )

    def test_build_transformer(self):
        transformer = build_transformer(self._feature_params)
        bad_df = self._df.copy()
        bad_df.loc[[0, 10, 50], 'chol'] = np.nan
        bad_df.loc[[10, 20, 40], 'fbs'] = np.nan
        actual = transformer.fit_transform(bad_df)
        self.assertFalse(np.isnan(actual).any())
        one_hot_columns_num = self._df[self._categorical_features].nunique().sum()
        self.assertEqual(actual.shape[1], one_hot_columns_num + len(self._numerical_features))
        self.assertEqual(actual[0, one_hot_columns_num + 1], bad_df.chol.mean())

    def test_extract_target(self):
        expected = self._df[self._target_col]
        actual = extract_target(self._df, self._feature_params)
        pd.testing.assert_series_equal(actual, expected)
