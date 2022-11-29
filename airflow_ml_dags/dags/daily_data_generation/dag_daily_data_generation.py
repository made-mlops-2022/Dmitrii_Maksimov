import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

from dags.daily_data_generation.scripts.data_generation import get_and_save_data

from configs.global_config import DATA_PATH, TARGET_PATH, DEFAULT_ARGS


dag_kwargs = {
    'dag_id': 'dag_data_generation',
    'description': 'Generate data and target',
    'schedule_interval': '00 21 * * *',
    'start_date': pendulum.datetime(2022, 11, 28, tz="CET"),
    'catchup': False,
    'default_args': DEFAULT_ARGS
}

with DAG(**dag_kwargs) as dag:
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    generate_data = PythonOperator(
        task_id='get_and_save_data',
        python_callable=get_and_save_data,
        op_kwargs={
            'data_path': DATA_PATH,
            'target_path': TARGET_PATH
        }
    )

    start >> generate_data >> end
