import os
import shutil
import tempfile
from activate import Session
from helper import invalidate_annotator
import json
from db import (
    Dataset, DatasetImage, ImageStatus, ModelVersion,
    PredictionBox, TrainingConfig, BoundingBoxClass, TrainingJobImage
)
from ml_tracking import log_training_run
from PIL import Image


TRAINABLE_STATUSES = {"labeled", "ready_for_training", "finalized"}


def _load_dataset_class_names(dataset_id: int) -> list[str]:
    with Session() as session:
        dataset_row = session.get(Dataset, dataset_id)
        if not dataset_row:
            raise RuntimeError(f"dataset with id={dataset_id} not found")

        classes = (
            session.query(BoundingBoxClass)
            .filter(BoundingBoxClass.dataset_id == dataset_id)
            .order_by(BoundingBoxClass.class_id)
            .all()
        )

        if not classes:
            raise RuntimeError("для датасета не задан список классов")

        max_class_id = max(c.class_id for c in classes)
        class_names = ["unknown"] * (max_class_id + 1)
        for cls in classes:
            class_names[cls.class_id] = cls.name

        return class_names


def _get_trainable_images(dataset_id: int, incremental: bool = False) -> list[DatasetImage]:
    """
    Возвращает изображения с нужными статусами.
    Если incremental=True — только те, которые ещё не участвовали ни в одном обучении.
    """
    with Session() as session:
        statuses = (
            session.query(ImageStatus)
            .filter(ImageStatus.code.in_(TRAINABLE_STATUSES))
            .all()
        )
        status_ids = {s.id for s in statuses}

        query = session.query(DatasetImage).filter(
            DatasetImage.dataset_id == dataset_id,
            DatasetImage.status_id.in_(status_ids),
        )

        if incremental:
            # исключаем те, что уже есть в training_job_image
            already_trained_ids = session.query(
                TrainingJobImage.dataset_image_id
            ).filter(
                TrainingJobImage.dataset_image_id == DatasetImage.id
            ).scalar_subquery()

            query = query.filter(
                ~DatasetImage.id.in_(
                    session.query(TrainingJobImage.dataset_image_id)
                    .filter(
                        TrainingJobImage.dataset_image_id == DatasetImage.id
                    )
                )
            )

        images = query.all()
        for img in images:
            session.expunge(img)
        return images


def _get_boxes_for_image(image_id: int) -> list[PredictionBox]:
    with Session() as session:
        boxes = (
            session.query(PredictionBox)
            .filter(PredictionBox.dataset_image_id == image_id)
            .all()
        )
        for b in boxes:
            session.expunge(b)
        return boxes


def _build_temp_dataset(dataset_path: str, images: list[DatasetImage]) -> str:
    """
    Строит временную папку со структурой YOLO:
      tmp/
        images/  — симлинки на оригинальные файлы
        labels/  — txt из БД
    """
    tmp_dir = tempfile.mkdtemp(prefix="yolo_train_")
    images_dir = os.path.join(tmp_dir, "images")
    labels_dir = os.path.join(tmp_dir, "labels")
    os.makedirs(images_dir)
    os.makedirs(labels_dir)

    for img in images:
        src = os.path.join(dataset_path, "images", img.filename)
        if not os.path.exists(src):
            continue

        dst = os.path.join(images_dir, img.filename)
        os.symlink(src, dst)

        boxes = _get_boxes_for_image(img.id)
        if not boxes:
            continue

        with Image.open(src) as pil_img:
            img_w, img_h = pil_img.size

        label_lines = []
        for box in boxes:
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                continue
            if box.x1 < 0 or box.y1 < 0 or box.x2 > img_w or box.y2 > img_h:
                continue

            x_center = ((box.x1 + box.x2) / 2) / img_w
            y_center = ((box.y1 + box.y2) / 2) / img_h
            width = (box.x2 - box.x1) / img_w
            height = (box.y2 - box.y1) / img_h

            if any(v > 1.0 for v in [x_center, y_center, width, height]):
                continue

            label_lines.append(
                f"{box.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if label_lines:
            stem = os.path.splitext(img.filename)[0]
            with open(os.path.join(labels_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(label_lines))

    return tmp_dir


def _record_trained_images(job_id: int, images: list[DatasetImage]):
    """Записывает в TrainingJobImage какие изображения использовались в этом джобе."""
    with Session() as session:
        for img in images:
            session.add(TrainingJobImage(
                training_job_id=job_id,
                dataset_image_id=img.id,
            ))
        session.commit()


def _train_and_save(
    dataset: Dataset,
    annotator,
    activate_new_version: bool = True,
    architecture_override: str | None = None,
    job_id: int | None = None,
    incremental: bool = False,
) -> tuple[str, int, int]:
    """
    job_id     — если передан, записывает использованные изображения в TrainingJobImage
    incremental — если True, берёт только изображения не участвовавшие в предыдущих джобах
    """

    with Session() as session:
        config = (
            session.query(TrainingConfig)
            .filter(TrainingConfig.dataset_id == dataset.id)
            .first()
        )
        last_version = (
            session.query(ModelVersion)
            .filter(
                ModelVersion.dataset_id == dataset.id,
                ModelVersion.is_active == True,
            )
            .first()
        )
        last_map = last_version.map50_95 if last_version else None

    class_names = _load_dataset_class_names(dataset.id)

    trainable_images = _get_trainable_images(dataset.id, incremental=incremental)
    if not trainable_images:
        raise RuntimeError(
            "нет новых изображений для обучения"
            if incremental
            else "нет изображений с подходящим статусом для обучения"
        )

    tmp_dir = _build_temp_dataset(dataset.path, trainable_images)

    use_augment = (
        config.augmentation_enabled
        and last_map is not None
        and last_map < config.augmentation_threshold
    ) if config else False

    try:
        if config:
            model_path, version = annotator.train(
                dataset_path=tmp_dir,
                class_names=class_names,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                imgsz=config.imgsz,
                optimizer=config.optimizer,
                augment=use_augment,
                save_dir=dataset.path
            )
            used_epochs = config.epochs
        else:
            model_path, version = annotator.train(
                dataset_path=tmp_dir,
                class_names=class_names,
                augment=use_augment,
                save_dir=dataset.path,
            )
            used_epochs = 10

        metrics = annotator.evaluate(dataset_path=tmp_dir, class_names=class_names)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # записываем какие изображения обучали
    if job_id is not None:
        _record_trained_images(job_id, trainable_images)

    run_id = log_training_run(
        dataset_name=dataset.name,
        architecture=architecture_override or dataset.current_model_architecture,
        hyperparams={
            "epochs": config.epochs if config else used_epochs,
            "learning_rate": config.learning_rate if config else 0.001,
            "batch_size": config.batch_size if config else 16,
            "imgsz": config.imgsz if config else 640,
            "optimizer": config.optimizer if config else "AdamW",
        },
        metrics={
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "mean_iou": metrics["mean_iou"],
        },
        model_path=model_path,
    )

    with Session() as session:
        if activate_new_version:
            session.query(ModelVersion).filter(
                ModelVersion.dataset_id == dataset.id,
                ModelVersion.is_active == True,
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
            mlflow_run_id=run_id,
        )
        session.add(model_record)
        session.commit()
        session.refresh(model_record)
        model_record_id = model_record.id

    if activate_new_version:
        invalidate_annotator(dataset.id)

    return model_path, version, model_record_id