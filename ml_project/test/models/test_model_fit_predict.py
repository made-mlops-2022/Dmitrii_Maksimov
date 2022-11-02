from test.preparations import generate_dataset, get_training_params

import unittest
from unittest.mock import MagicMock, patch

import copy
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.models.model_fit_predict import get_model, predict_model, evaluate_model


class TestModelFitPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._data_path = 'data/test/test.csv'
        cls._df = generate_dataset(cls._data_path)
        cls._training_params = get_training_params()

    def test_get_model(self):
        svm_model = get_model(self._training_params)
        new_params = copy.copy(self._training_params)
        new_params.model_type = 'RandomForestClassifier'
        rf_model = get_model(new_params)
        self.assertTrue(isinstance(svm_model, SVC))
        self.assertTrue(isinstance(rf_model, RandomForestClassifier))

    def test_predict_model(self):
        m_model = MagicMock()
        m_df = MagicMock()
        predict_model(m_model, m_df)
        m_model.predict.assert_called_once_with(m_df)

    @patch('src.models.model_fit_predict.f1_score')
    @patch('src.models.model_fit_predict.recall_score')
    @patch('src.models.model_fit_predict.precision_score')
    @patch('src.models.model_fit_predict.accuracy_score')
    def test_evaluate_model(self, m_acc, m_pr, m_rec, m_f1):
        m_predicts = MagicMock()
        m_target = MagicMock()
        evaluate_model(m_predicts, m_target)

        m_acc.assert_called_once_with(m_target, m_predicts)
        m_pr.assert_called_once_with(m_target, m_predicts)
        m_rec.assert_called_once_with(m_target, m_predicts)
        m_f1.assert_called_once_with(m_target, m_predicts)
