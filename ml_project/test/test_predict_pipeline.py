from test.preparations import generate_dataset, get_predict_pipeline_params

import unittest
from unittest.mock import MagicMock, patch, ANY

from src.predict_pipeline import predict_pipeline


class TestModelFitPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._data_path = 'data/test/test.csv'
        cls._df = generate_dataset(cls._data_path)
        cls._training_pipeline_params = get_predict_pipeline_params('configs/predict/predict_config.yaml')
        cls._training_pipeline_params.input_data_path = cls._data_path

    @patch('src.predict_pipeline.open')
    @patch('src.predict_pipeline.pd')
    @patch('src.predict_pipeline.predict_model')
    @patch('src.predict_pipeline.pickle')
    @patch('src.predict_pipeline.logger')
    def test_predict_pipeline(self, m_logger, m_pickle, m_predict_model, m_pd, m_open):
        file = MagicMock()
        m_open.return_value = file
        model = MagicMock()
        predict = MagicMock()
        df = MagicMock()
        m_pickle.load.return_value = model
        m_predict_model.return_value = predict
        m_pd.DataFrame.return_value = df
        predict_pipeline(self._training_pipeline_params)
        m_logger.info.assert_called()
        m_pickle.load.assert_called_once_with(file)
        m_predict_model.assert_called_once_with(model, ANY)
