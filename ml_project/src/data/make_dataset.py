from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import requests

from src.entities.split_params import SplittingParams


def download_dataset():
    URL = ('https://storage.googleapis.com/kagglesdsdata/datasets/576697/1043970/'
           'heart_cleveland_upload.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential'
           '=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221030%2Fauto%2Fsto'
           'rage%2Fgoog4_request&X-Goog-Date=20221030T094016Z&X-Goog-Expires=259200&X-Goog-'
           'SignedHeaders=host&X-Goog-Signature=0e157e3f5f2f3616b90825a001e083ffd29b88d91911a'
           'e90d727e74e529c3bb3f854d904e73faa897a762dddf7e30b85868476eddfdad267cdabd19c8ad65b2f2'
           'b037cac7815a2e9a7f57e3957d088ffa42d8cff57e98778453b6a226ea43badac27e37fa5dda92c7b8a170'
           '70d92d20cb7999316fea532c40507e4efca0d6ac828797043b824a023289c844623ee03f129ee77f495dd01'
           'ec8b6f423214fa1913a6e9b4f5a04def644e25328c335744ad4377baf5c34eed6769810d641aa6b93d03570d'
           '9ce4be4ae0a7b62ee9f6e711375668f02da6f8aad2c6e292246baad8e3f188691f79fcb805548b4cf063d70ff2a0c48d34e679d2c04eb00632a35d40c2'
           )
    OUTPUT_PATH = 'data/raw/heart_cleveland_upload.csv'
    r = requests.get(URL)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(r.text)


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(data, test_size=params.val_size, random_state=params.random_state)
    return train_data, val_data
