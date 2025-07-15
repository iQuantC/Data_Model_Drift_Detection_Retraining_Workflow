from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from datetime import datetime
import subprocess


def decide_retrain_branch():
    """
    Run the drift detection script and decide whether to retrain.
    """
    result = subprocess.run(
        ["python", "drift_detection/check_drift.py"],
        capture_output=True,
        text=True
    )
    output = result.stdout
    print("Drift detection output:\n", output)

    # Determine drift based on script output
    if "Drift detected!" in output:
        return "retrain_model"
    else:
        return "no_retrain"


def retrain_model_task():
    """
    Retrain the model.
    """
    print("Starting retraining process...")
    subprocess.run(["python", "models/train_model.py"])
    print("Retraining completed.")


def skip_retrain_task():
    """
    Skip retraining.
    """
    print("No drift detected. Skipping retraining.")


with DAG(
    dag_id="data_model_drift_detection_and_retraining",
    schedule=None,  # Manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description="Detect model drift and retrain if needed",
    tags=["ml", "ai", "mlops", "drift detection"]
) as dag:

    decide_branch = BranchPythonOperator(
        task_id="decide_retrain_branch",
        python_callable=decide_retrain_branch,
    )

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model_task,
    )

    no_retrain = PythonOperator(
        task_id="no_retrain",
        python_callable=skip_retrain_task,
    )

    decide_branch >> [retrain_model, no_retrain]