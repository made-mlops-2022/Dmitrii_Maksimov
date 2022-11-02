from test.preparations import generate_dataset, get_training_pipeline_params

import unittest
from unittest.mock import MagicMock, patch, ANY

from src.train_pipeline import train_pipeline


class TestModelFitPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._data_path = 'data/test/test.csv'
        cls._df = generate_dataset(cls._data_path)
        cls._training_pipeline_params = get_training_pipeline_params('configs/train/train_config_rf.yaml')
        cls._training_pipeline_params.input_data_path = cls._data_path

    @patch('src.train_pipeline.open')
    @patch('src.train_pipeline.serialize_model')
    @patch('src.train_pipeline.json')
    @patch('src.train_pipeline.logger')
    def test_train_pipeline(self, m_logger, m_json, m_serialize_model, m_open):
        file = MagicMock()
        m_open.return_value.__enter__.return_value = file
        train_pipeline(self._training_pipeline_params)
        m_json.dump.assert_called_once_with(ANY, file)
        m_logger.info.assert_called()
        m_serialize_model.assert_called()
