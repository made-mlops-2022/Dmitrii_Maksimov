from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_predict_disease_200():
    request = {
        "age": 30,
        "sex": 1,
        "cp": 0,
        "trestbps": 100,
        "chol": 200,
        "fbs": 0,
        "restecg": 2,
        "thalach": 100,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 1,
        "ca": 0,
        "thal": 1,
    }
    response = client.post("/predict", json=request)
    assert response.status_code == 200
    assert response.json() == [{"condition": 0}] or response.json() == [{"condition": 1}]


def test_predict_400():
    request = {
        "age": 210,
        "sex": 1,
        "cp": 0,
        "trestbps": 100,
        "chol": 200,
        "fbs": 0,
        "restecg": 2,
        "thalach": 100,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 1,
        "ca": 0,
        "thal": 1,
    }
    response = client.post("/predict", json=request)
    assert response.status_code == 400
    assert response.json()["detail"][0]["msg"] == "ValueError: age value 210.0"
