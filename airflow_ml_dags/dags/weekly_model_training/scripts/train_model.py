import logging

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from dags.weekly_model_training.config import TrainingPipelineParams, FeatureParams, TrainingParams, replace_ds_nodas

def train_model(params: TrainingPipelineParams, **kwargs):
    params.output_model_path = replace_ds_nodas(kwargs['ds_nodash'], params.output_model_path)
    params.splitting_params.train_path = replace_ds_nodas(kwargs['ds_nodash'], params.splitting_params.train_path)
    logging.info(f"start train pipeline with params {params}")
    data = pd.read_csv(params.splitting_params.train_path)
    logging.info(f"data.shape is {data.shape}")

    transformer = _build_transformer(params.feature_params)
    model = _get_model(transformer, params.train_params)
    train_target = _extract_target(data, params.feature_params)

    model.fit(data, train_target)

    _serialize_model(model, params.output_model_path)


def _build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def _build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean"))]
    )
    return num_pipeline


def _build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                _build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                _build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def _extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target


def _get_model(transformer: ColumnTransformer, train_params: TrainingParams) -> Pipeline:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, random_state=train_params.random_state)
    elif train_params.model_type == "SupportVectorMachine":
        model = SVC(random_state=train_params.random_state)
    else:
        raise NotImplementedError()
    return Pipeline([
        ("transformer", transformer),
        ("model", model)
        ])


def _serialize_model(model: Pipeline, output: str) -> None:
    dirname = os.path.dirname(output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
