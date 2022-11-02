import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.entities.train_params import TrainingParams

SklearnClassifierModel = Union[RandomForestClassifier, SVC]


def get_model(train_params: TrainingParams) -> SklearnClassifierModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, random_state=train_params.random_state)
    elif train_params.model_type == "SupportVectorMachine":
        model = SVC(random_state=train_params.random_state)
    else:
        raise NotImplementedError()
    return model


def predict_model(model: SklearnClassifierModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "accuracy_score": accuracy_score(target, predicts),
        "precision_score": precision_score(target, predicts),
        "recall_score": recall_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def serialize_model(model: SklearnClassifierModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output