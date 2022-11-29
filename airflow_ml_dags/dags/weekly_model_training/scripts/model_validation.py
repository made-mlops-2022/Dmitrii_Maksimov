import logging
from typing import Dict

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json

from configs.global_config import TARGET


def validate_model(model_path: str, data_path: str, metrics_path: str):
    df = pd.read_csv(data_path)
    logging.info(f"Validation is {df.shape}")

    model = pickle.load(open(model_path, 'rb'))
    logging.info(f"Model is loaded")
    data = df.drop(columns=TARGET)
    target = df[TARGET]
    predicts = model.predict(data)
    logging.info(f"Validation size: {len(predicts)}")
    metrics = _evaluate_model(predicts, target)
    _save_metrics(metrics, metrics_path)


def _evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "accuracy_score": accuracy_score(target, predicts),
        "precision_score": precision_score(target, predicts),
        "recall_score": recall_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def _save_metrics(metrics: Dict[str, float], path: str) -> None:
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(path, "w") as metric_file:
        json.dump(metrics, metric_file)
