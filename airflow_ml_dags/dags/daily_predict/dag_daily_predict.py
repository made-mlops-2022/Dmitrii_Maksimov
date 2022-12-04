import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable

from dags.daily_predict.scripts.prediction import predict

from configs.global_config import DATA_PATH, PREDICTION_PATH, DEFAULT_ARGS


dag_kwargs = {
    'dag_id': 'dag_daily_predict',
    'description': 'Predict data',
    'schedule_interval': '00 10 * * *',
    'start_date': pendulum.datetime(2022, 11, 28, tz="CET"),
    'catchup': False,
    'default_args': DEFAULT_ARGS
}

model_path = Variable.get("model_path")

with DAG(**dag_kwargs) as dag:
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')

    predict_data = PythonOperator(
        task_id='predict_data',
        python_callable=predict,
        op_kwargs={
            'model_path': model_path,
            'data_path': DATA_PATH,
            'result_path': PREDICTION_PATH
        }
    )

    start >> predict_data >> end
