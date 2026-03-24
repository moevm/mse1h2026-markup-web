from activate import Session
from helper import invalidate_annotator
import json
from db import Dataset, ModelVersion, TrainingConfig
from ml_tracking import log_training_run

def _train_and_save(dataset: Dataset, dataset_name: str, annotator) -> tuple[str, int]:
    '''общий блок: достаём гиперпараметры, обучаем, считаем метрики, сохраняем версию'''

    # достаём гиперпараметры из бд
    with Session() as session:
        config = session.query(TrainingConfig).filter(
            TrainingConfig.dataset_id == dataset.id
        ).first()

    if config:
        model_path, version = annotator.train(
            dataset_name,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            imgsz=config.imgsz,
            optimizer=config.optimizer
        )
        used_epochs = config.epochs
    else:
        model_path, version = annotator.train(dataset_name)
        used_epochs = 10

    # валидация —> считаем метрики
    metrics = annotator.evaluate(dataset_name)

    run_id = log_training_run(
        dataset_name=dataset_name,
        architecture=dataset.current_model_architecture,
        hyperparams={
            "epochs": config.epochs if config else used_epochs,
            "learning_rate": config.learning_rate if config else 0.001,
            "batch_size": config.batch_size if config else 16,
            "imgsz": config.imgsz if config else 640,
            "optimizer": config.optimizer if config else "AdamW"
        },
        metrics={
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "mean_iou": metrics["mean_iou"]
        },
        model_path=model_path
    )

    with Session() as session:
        # деактивируем старые версии для этого датасета
        session.query(ModelVersion).filter(
            ModelVersion.dataset_id == dataset.id,
            ModelVersion.is_active == True
        ).update({"is_active": False})

        model_record = ModelVersion(
            dataset_id=dataset.id,
            version=version,
            path=model_path,
            epochs=used_epochs,
            is_active=True,
            architecture=dataset.current_model_architecture,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            map50=metrics["map50"],
            map50_95=metrics["map50_95"],
            mean_iou=metrics["mean_iou"],
            confusion_matrix_json=json.dumps(metrics["confusion_matrix"]),
            mlflow_run_id = run_id
        )
        session.add(model_record)
        session.commit()

    invalidate_annotator(dataset.id)
    return model_path, version