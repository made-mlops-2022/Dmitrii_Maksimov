
import pandas as pd
import requests
import json
import os


if __name__ == "__main__":
    data = pd.read_csv(os.getenv("PATH_TO_DATA"))
    data.drop("condition", axis=1, inplace=True)

    for row in data.to_dict(orient="records"):
        response = requests.post("http://0.0.0.0:8000/predict", json.dumps(row))
        print(response.status_code)
        print(response.json())
