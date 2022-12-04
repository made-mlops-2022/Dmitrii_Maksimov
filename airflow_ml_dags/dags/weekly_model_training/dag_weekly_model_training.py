import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.contrib.sensors.file_sensor import FileSensor

from dags.weekly_model_training.scripts.data_preparation import prepare_dataset
from dags.weekly_model_training.scripts.data_splitting import split_data
from dags.weekly_model_training.scripts.train_model import train_model
from dags.weekly_model_training.scripts.model_validation import validate_model
from dags.weekly_model_training.config import read_training_pipeline_params

from configs.global_config import DATA_PATH, TARGET_PATH, ML_CONFIG_PATH, DEFAULT_ARGS


dag_kwargs = {
    'dag_id': 'dag_weekly_model_training',
    'description': 'Train and evaluate model',
    'schedule_interval': '30 21 * * *',
    'start_date': pendulum.datetime(2022, 11, 28, tz="CET"),
    'catchup': False,
    'default_args': DEFAULT_ARGS
}

params = read_training_pipeline_params(ML_CONFIG_PATH)

with DAG(**dag_kwargs) as dag:
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    check_data = FileSensor(
        task_id="check_data",
        poke_interval=30,
        timeout=60 * 60,
        filepath=DATA_PATH
    )

    check_target = FileSensor(
        task_id="check_target",
        poke_interval=30,
        timeout=60 * 60,
        filepath=TARGET_PATH
    )

    prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_dataset,
        op_kwargs={
            'data_path': DATA_PATH,
            'target_path': TARGET_PATH,
            'dataset_path': params.input_data_path,
        }
    )

    train_val_split = PythonOperator(
        task_id='train_val_split',
        python_callable=split_data,
        provide_context=True,
        op_kwargs={
            'dataset_path': params.input_data_path,
            'params': params.splitting_params,
        }
    )

    train_and_save_model = PythonOperator(
        task_id='train_and_save_model',
        python_callable=train_model,
        provide_context=True,
        op_kwargs={
            'params': params,
        }
    )

    val_model = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        op_kwargs={
            'model_path': params.output_model_path,
            'data_path': params.splitting_params.val_path,
            'metrics_path': params.metric_path
        }
    )

    start >> [check_data, check_target] >> prepare_data >> train_val_split >> train_and_save_model >> val_model >> end
