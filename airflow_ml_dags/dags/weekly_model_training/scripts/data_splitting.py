from sklearn.model_selection import train_test_split
import pandas as pd
import os
import logging

from dags.weekly_model_training.config import SplittingParams
from dags.weekly_model_training.config import replace_ds_nodas


def split_data(dataset_path: str, params: SplittingParams, **kwargs):
    params.train_path = replace_ds_nodas(kwargs['ds_nodash'], params.train_path)
    params.val_path = replace_ds_nodas(kwargs['ds_nodash'], params.val_path)
    df = pd.read_csv(dataset_path)
    logging.info(f'Read dataset: {df.shape}')
    train_data, val_data = train_test_split(df, test_size=params.val_size, random_state=params.random_state)
    logging.info(f'Split dataset: {train_data.shape=}, {val_data.shape=}')
    _save_data(train_data, params.train_path)
    logging.info(f'Train data was saved: {params.train_path}')
    _save_data(val_data, params.val_path)
    logging.info(f'Validation data was saved: {params.val_path}')


def _save_data(data: pd.DataFrame, data_path: str) -> None:
    dirname = os.path.dirname(data_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    data.to_csv(data_path, index=False)
