FROM python:3.8-slim-buster
COPY requirements.txt app.py validator_app.py .env ./online_inference/

WORKDIR /online_inference

RUN python -m pip install --upgrade pip \
    pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
