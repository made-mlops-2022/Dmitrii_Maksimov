import os
import pickle
from typing import List, Optional
from boto3 import client
import pandas as pd
import uvicorn
from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

from validator_app import HeartDiseaseModel, HeartDiseaseResponse


def load_object(path: str) -> Pipeline:
    MODEL_FILE = 'model.pkl'
    s3 = client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    s3.download_file(os.getenv('BUCKET_TO_MODEL'), path, MODEL_FILE)
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


model: Optional[Pipeline] = None


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    load_dotenv()
    model_path = os.getenv("PATH_TO_MODEL")
    model = load_object(model_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.post("/predict", response_model=List[HeartDiseaseResponse])
def predict(data: HeartDiseaseModel):
    input_df = pd.DataFrame([data.dict()])
    predicts = model.predict(input_df)
    return list(
        HeartDiseaseResponse(condition=result) for result in predicts
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
