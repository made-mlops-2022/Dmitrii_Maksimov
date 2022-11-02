from src.entities.split_params import SplittingParams
from src.entities.feature_params import FeatureParams
from src.entities.train_params import TrainingParams
from src.entities.train_pipeline_params import read_training_pipeline_params
from src.entities.predict_pipeline_params import read_predict_pipeline_params

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def generate_dataset(path: str) -> pd.DataFrame:
    SIZE = 100
    data = {}
    data["sex"] = pd.Series(np.random.randint(0, 2, size=SIZE))
    data["cp"] = pd.Series(np.random.randint(0, 4, size=SIZE))
    data["fbs"] = pd.Series(np.random.randint(0, 2, size=SIZE))
    data["restecg"] = pd.Series(2 * np.random.randint(0, 2, size=SIZE))
    data["exang"] = pd.Series(np.random.randint(0, 2, size=SIZE))
    data["slope"] = pd.Series(np.random.randint(0, 3, size=SIZE))
    data["ca"] = pd.Series(np.random.randint(0, 3, size=SIZE))
    data["thal"] = pd.Series(np.random.randint(0, 3, size=SIZE))
    data["age"] = pd.Series(np.random.randint(18, 91, size=SIZE))
    data["trestbps"] = pd.Series(np.random.randint(94, 201, size=SIZE))
    data["chol"] = pd.Series(np.random.randint(126, 565, size=SIZE))
    data["thalach"] = pd.Series(np.random.randint(71, 203, size=SIZE))
    data["oldpeak"] = pd.Series(6.2 * np.random.random(size=SIZE))
    data["condition"] = pd.Series(np.random.randint(0, 2, size=SIZE))
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def get_splitting_params(**kwargs):
    return SplittingParams(**kwargs)

def get_feature_params(**kwargs):
    return FeatureParams(**kwargs)

def get_training_params(**kwargs):
    return TrainingParams(**kwargs)

def get_training_pipeline_params(path: str):
    return read_training_pipeline_params(yaml.safe_load(Path(path).read_text()))

def get_predict_pipeline_params(path: str):
    return read_predict_pipeline_params(yaml.safe_load(Path(path).read_text()))
