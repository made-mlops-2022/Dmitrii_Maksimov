import pandas as pd
import os

from typing import Tuple, Union
import logging

from configs.global_config import TARGET, URL


def get_and_save_data(data_path: str, target_path: str):
    data, target = _read_data()
    _save_data(data, data_path)
    logging.info(f'Data {data.shape} was saved in {data_path}')
    _save_data(target, target_path)
    logging.info(f'Target {target.shape} was saved in {target_path}')


def _read_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(URL)
    data = df.drop(columns=TARGET)
    target = df[TARGET]
    return data, target


def _save_data(data: Union[pd.DataFrame, pd.Series], data_path: str) -> None:
    dirname = os.path.dirname(data_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    data.to_csv(data_path, index=False)
