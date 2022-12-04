import pandas as pd
import os

from typing import Tuple, Union
import logging

import click


@click.command("get_and_save_data")
@click.option('--data_path')
@click.option('--target_path')
@click.option('--target_col')
@click.option('--url')
def get_and_save_data(data_path: str, target_path: str, target_col: str, url: str):
    data, target = _read_data(target_col, url)
    _save_data(data, data_path)
    logging.info(f'Data {data.shape} was saved in {data_path}')
    _save_data(target, target_path)
    logging.info(f'Target {target.shape} was saved in {target_path}')


def _read_data(target_col: str, url: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(url)
    data = df.drop(columns=target_col)
    target = df[target_col]
    return data, target


def _save_data(data: Union[pd.DataFrame, pd.Series], data_path: str) -> None:
    dirname = os.path.dirname(data_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    data.to_csv(data_path, index=False)


if __name__ == '__main__':
    get_and_save_data()
