from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    'test_dag',
    default_args={'owner': 'airflow'},
    schedule_interval='@daily',
    start_date=datetime(2025, 10, 1),
    catchup=False,
) as dag:

    hello = BashOperator(
        task_id='say_hello',
        bash_command='echo "Hello Airflow Pipeline Works!"'
    )
