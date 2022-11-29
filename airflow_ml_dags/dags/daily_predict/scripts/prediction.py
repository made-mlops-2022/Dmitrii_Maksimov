import logging

import pandas as pd
import pickle
import os


def predict(model_path: str, data_path: str, result_path: str):
    data = pd.read_csv(data_path)
    logging.info(f"Predict is {data.shape}")

    model = pickle.load(open(model_path, 'rb'))
    logging.info(f"Model is loaded")
    predicts = model.predict(data)
    _save_prediction(pd.Series(predicts), result_path)
    logging.info(f"predictions are saved in {result_path}")


def _save_prediction(data: pd.Series, data_path: str) -> None:
    dirname = os.path.dirname(data_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    data.to_csv(data_path, index=False)
