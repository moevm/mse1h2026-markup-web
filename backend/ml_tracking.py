import mlflow


def log_training_run(dataset_name, architecture, hyperparams, metrics, model_path):
    """логирует обучение в mlflow, возвращает run_id"""

    mlflow.set_experiment(f'dataset: {dataset_name}')

    with mlflow.start_run() as run:
        mlflow.log_params({
            "architecture": architecture,
            **hyperparams
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_path)

        return run.info.run_id
    