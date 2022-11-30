airflow_ml_dags
==============================

3_Homewerok on MLOPS at the MADE

# Usage
1. If you want to get notifications do following:
    - in configs/global_config.py:
        - set up email on which you want to get emails
    - in docker-compose.yml:
        - set up SMTP data:
            ~~~
            AIRFLOW__SMTP__SMTP_HOST=
            AIRFLOW__SMTP__SMTP_PORT=
            AIRFLOW__SMTP__SMTP_USER=
            AIRFLOW__SMTP__SMTP_PASSWORD=
            AIRFLOW__SMTP__SMTP_MAIL_FROM=
            ~~~
2. Build image:
~~~
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build
~~~
3. Choose model for prediction:
in docker-compose.yml:
~~~
AIRFLOW_VAR_MODEL_PATH=data/models/20221129/model.pkl
~~~
