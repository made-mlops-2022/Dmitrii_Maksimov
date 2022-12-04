from typing import Tuple
import logging

import pandas as pd
import os

from configs.global_config import TARGET


def prepare_dataset(data_path: str, target_path: str, dataset_path: str):
    logging.info(f'Create dataset using {data_path=}, {target_path=}')
    dataset = _get_dataset(data_path, target_path)
    logging.info(f'Create dataset: {dataset.shape}, {dataset.columns}')
    _save_data(dataset, dataset_path)
    logging.info(f'Dataset was saved in {dataset_path}')


def _get_dataset(data_path: str, target_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    target = pd.read_csv(target_path)
    data[TARGET] = target
    return data


def _save_data(data: pd.DataFrame, data_path: str) -> None:
    dirname = os.path.dirname(data_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    data.to_csv(data_path, index=False)
