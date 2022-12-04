from datetime import timedelta


DEFAULT_ARGS = {
    'owner': 'maksimov_dmitrii',
    'email': ['maksimov.dmitry.m@gmail.com'],
    'retries': 5,
    'retry_delay': timedelta(minutes=1),
    'email_on_failure': True,
    'email_on_retry': False
}

DATA_PATH = 'data/raw/{{ ds_nodash }}/data.csv'
TARGET_PATH = 'data/raw/{{ ds_nodash }}/target.csv'
PREDICTION_PATH = 'data/predictions/{{ ds_nodash }}/predictions.csv'
TARGET = 'condition'
URL = 'https://storage.googleapis.com/kagglesdsdata/datasets/576697/1043970/heart_cleveland_upload.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221129%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221129T133749Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=21c6ca0586d32d9d5137a3f9bab3aa181bcd39feacf7732073279fdd2e637de8d08e160da90ccde60a7e3dbb5889732a34fd6b56c0001722fbec17bb60abd831bdd2522a8abfa73ec67b2f7aaae6c026da78acfd0de72e1f2a5c2c86d7182ae64c24e408a2b33d825da1409f0f0fe021cbc16f9e658291f9317d5447418779a78c40b8197328b36e490139abd282a49c2152ea67b9661bf8296a36fea62726f59ad993431ab1b4d83d5f9ebf55ef4e7a75a61a16b50f6133ceddc04d59ec71e10b1fcb4562b2d4115d64c0800fc75cc09ba72379905b1014a78c3180d7e6a87d2836e8e3d3c18b9825968d188417a159ecf3ed0985adfff55cdb1bc6ca7706c0'
ML_CONFIG_PATH = 'configs/ml_config.yaml'
