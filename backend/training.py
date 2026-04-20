from activate import Session
from helper import invalidate_annotator
import json
from db import Dataset, ModelVersion, TrainingConfig
from ml_tracking import log_training_run


def _load_dataset_class_names(dataset_id: int) -> list[str]:
    with Session() as session:
        dataset_row = session.get(Dataset, dataset_id)
        if not dataset_row:
            raise RuntimeError(f"dataset with id={dataset_id} not found")
        classes_json = dataset_row.classes_json

    if not classes_json:
        raise RuntimeError("для датасета не задан список классов")

    try:
        class_names = json.loads(classes_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("для датасета задан некорректный classes_json") from exc

    if not isinstance(class_names, list) or len(class_names) == 0:
        raise RuntimeError("список классов датасета пуст или имеет неверный формат")

    for class_name in class_names:
        if not isinstance(class_name, str) or class_name.strip() == "":
            raise RuntimeError("список классов датасета содержит некорректные значения")

    return class_names


def _train_and_save(
    dataset: Dataset,
    annotator,
    activate_new_version: bool = True,
    architecture_override: str | None = None
) -> tuple[str, int, int]:
    '''общий блок: достаём гиперпараметры, обучаем, считаем метрики, сохраняем версию'''

    # достаём гиперпараметры из бд
    with Session() as session:
        config = session.query(TrainingConfig).filter(
            TrainingConfig.dataset_id == dataset.id
        ).first()
        
        last_version = session.query(ModelVersion).filter(
            ModelVersion.dataset_id == dataset.id,
            ModelVersion.is_active == True
        ).first()
        last_map = last_version.map50_95 if last_version else 0.0

    class_names = _load_dataset_class_names(dataset.id)

    use_augment = (
        config.augmentation_enabled and
        last_map < config.augmentation_threshold
    ) if config else False


    if config:
        model_path, version = annotator.train(
            dataset_path=dataset.path,
            class_names=class_names,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            imgsz=config.imgsz,
            optimizer=config.optimizer,
            augment=use_augment
        )
        used_epochs = config.epochs

        
    else:
        model_path, version = annotator.train(
            dataset_path=dataset.path,
            class_names=class_names,
            augment=use_augment
        )
        used_epochs = 10

    # валидация —> считаем метрики
    metrics = annotator.evaluate(dataset_path=dataset.path, class_names=class_names)

    run_id = log_training_run(
        dataset_name=dataset.name,
        architecture=architecture_override or dataset.current_model_architecture,
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

    model_record_id = None
    with Session() as session:
        if activate_new_version:
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
            is_active=activate_new_version,
            architecture=architecture_override or dataset.current_model_architecture,
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
        session.refresh(model_record)
        model_record_id = model_record.id

    if activate_new_version:
        invalidate_annotator(dataset.id)

    return model_path, version, model_record_id
