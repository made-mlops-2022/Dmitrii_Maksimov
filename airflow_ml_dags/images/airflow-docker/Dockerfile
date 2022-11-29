ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.3.0-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

USER root
ENV PYTHONPATH=/home/airflow/.local/lib/python3.8/site-packages
ENV PYTHONPATH "${PYTHONPATH}:$AIRFLOW_HOME"
