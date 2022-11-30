import pendulum

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator

from docker.types import Mount

from configs.global_config import DATA_PATH, TARGET_PATH, TARGET, URL, DEFAULT_ARGS


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

    generate_data = DockerOperator(
        image="airflow-data-generation",
        command=f"--data_path {DATA_PATH} --target_path {TARGET_PATH} --target_col {TARGET} --url {URL}",
        network_mode="bridge",
        task_id="get_and_save_data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/Users/dmitriimaksimov/Desktop/GitHub/Dmitrii_Maksimov/airflow_ml_dags/data", target="/data", type='bind')]
    )

    start >> generate_data >> end
