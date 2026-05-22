from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy import select
import torch
from activate import Session
from db import (
    DatasetImage,
    Dataset,
    DatasetStatus,
    ImageStatus,
    ModelVersion,
    PredictionBatch,
    PredictionBox,
    TrainingConfig,
    TrainingJob,
    BoundingBoxClass,
)
from model_registry import get_all_models, get_model_by_id
import os
from typing import Optional
from job_runner import submit_training_job
from helper import (
    get_annotator,
    invalidate_annotator,
    save_auto_accepted_labels,
    complete_batch_and_update_dataset_metrics,
)
import json
from urllib.parse import quote
import random

DATASETS_ROOT_HOST = os.getenv("DATASETS_ROOT_HOST", "/")
DATASETS_ROOT_CONTAINER = os.getenv("DATASETS_ROOT_CONTAINER", "/mnt/host")


def resolve_container_path(user_path: str) -> str:
    user_path = user_path
    host_root = DATASETS_ROOT_HOST
    if not user_path.lower().startswith(host_root.lower()):
        raise ValueError(f"Путь должен находиться внутри {host_root}")
    relative = user_path.replace('\\','/')
    relative = relative[len(host_root):].lstrip("/")
    relative = os.path.join(DATASETS_ROOT_CONTAINER, relative) 
    return relative


def dataset_has_labels(dataset_path: str) -> bool:
    labels_dir = os.path.join(dataset_path, "labels")
    if not os.path.isdir(labels_dir):
        return False

    for _, _, files in os.walk(labels_dir):
        for filename in files:
            if filename.lower().endswith(".txt"):
                return True
    return False


def list_dataset_images(dataset_path: str) -> list[str]:
    images_dir = os.path.join(dataset_path, "images")
    if not os.path.isdir(images_dir):
        return []
    return sorted(
        [
            filename
            for filename in os.listdir(images_dir)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.isfile(os.path.join(images_dir, filename))
        ]
    )


def get_random_image_from_dataset(dataset_path: str) -> Optional[str]:
    """Get a random image filename from dataset directory"""
    # Check if images are in 'images' subdirectory
    images_dir = os.path.join(dataset_path, "images")
    if os.path.isdir(images_dir):
        images = list_dataset_images(dataset_path)
        if images:
            return f"images/{random.choice(images)}"

    # Fallback to root directory
    images = list_dataset_images(dataset_path)
    if not images:
        return None
    return random.choice(images)


def select_batch_images_by_status(
    session, dataset_id: int, dataset_path: str, limit: int
) -> list[str]:
    image_filenames = list_dataset_images(dataset_path)
    if not image_filenames:
        return []

    existing_rows = (
        session.query(DatasetImage)
        .filter(
            DatasetImage.dataset_id == dataset_id,
            DatasetImage.filename.in_(image_filenames),
        )
        .all()
    )
    status_by_filename = {row.filename: row.status_id for row in existing_rows}
    eligible_statuses = {1, 4}

    selected = [
        filename
        for filename in image_filenames
        if filename not in status_by_filename
        or status_by_filename[filename] in eligible_statuses
    ]
    return selected[:limit]


router = APIRouter()


def _sync_dataset_status(session, dataset_id: int) -> None:
    """Пересчитать status_id, inwork_size и average_percent_success датасета по текущим статусам изображений."""
    images = session.query(DatasetImage).filter(
        DatasetImage.dataset_id == dataset_id
    ).all()
    if not images:
        return

    statuses = session.query(ImageStatus).all()
    unlabeled_ids = {s.id for s in statuses if s.code == "unlabeled"}
    done_ids = {s.id for s in statuses if s.code in ("labeled", "ready_for_training", "finalized")}

    total = len(images)
    unlabeled_count = sum(1 for img in images if img.status_id in unlabeled_ids)
    done_count = sum(1 for img in images if img.status_id in done_ids)

    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        return

    if unlabeled_count == total:
        dataset.status_id = 0  # Just load
    elif done_count == total:
        dataset.status_id = 1  # Done
    else:
        dataset.status_id = 3  # At work

    dataset.inwork_size = total - unlabeled_count
    dataset.average_percent_success = round(done_count / total * 100, 1) if total > 0 else 0


class AddDatasetRequest(BaseModel):
    dataset_name: str
    path: str


class ChangeModelRequest(BaseModel):
    architecture: str


class TrainingConfigRequest(BaseModel):
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    imgsz: int | None = None
    optimizer: str | None = None
    augmentation_enabled: bool | None = None
    augmentation_threshold: float | None = None
    auto_accept_enabled: bool | None = None
    auto_accept_confidence_threshold: float | None = None
    incremental_training_enabled: bool | None = None
    device: str | None = None


class AugmentationConfigRequest(BaseModel):
    augmentation_enabled: Optional[bool] = None
    augmentation_threshold: Optional[float] = None


class ClassItem(BaseModel):
    class_id: int
    name: str
    color: Optional[str] = None


class ClassesRequest(BaseModel):
    classes: list[ClassItem]



# дубликат, я не знаю какая ручка верная

# @router.get("/api/datasets/{dataset_id}/classes")
# async def get_dataset_classes(dataset_id: int):
#     with Session() as session:
#         classes = (
#             session.query(BoundingBoxClass)
#             .filter(BoundingBoxClass.dataset_id == dataset_id)
#             .order_by(BoundingBoxClass.class_id)
#             .all()
#         )
#         return [
#             {"class_id": c.class_id, "name": c.name, "color": c.color} for c in classes
#         ]


@router.get("/api/getDatasets")
async def get_datasets():
    """список всех датасетов в формате для фронта"""
    with Session() as session:
        datasets = session.query(Dataset).all()
        result = []
        for ds in datasets:
            status = (
                session.query(DatasetStatus)
                .filter(DatasetStatus.id == ds.status_id)
                .first()
            )

            # Get random image for preview
            random_image = get_random_image_from_dataset(ds.path)
            preview_url = None
            if random_image:
                # Create URL for serving the image
                preview_url = f"/api/datasets/{ds.id}/preview/{quote(random_image)}"

            result.append(
                {
                    "id": ds.id,
                    "name": ds.name,
                    "status_id": ds.status_id,
                    "status": {"id": status.id, "name": status.name},
                    "total_size": ds.total_size,
                    "inwork_size": ds.inwork_size,
                    "path": ds.path,
                    "preview_image": preview_url,
                    "average_percent_success": ds.average_percent_success,
                    "current_model_architecture": ds.current_model_architecture,
                    "metric_precision": ds.metric_precision,
                    "metric_recall": ds.metric_recall,
                    "metric_f1": ds.metric_f1,
                    "metric_mean_iou": ds.metric_mean_iou,
                    "metrics_boxes_total": ds.metrics_boxes_total,
                    "metrics_images_total": ds.metrics_images_total,
                    "lastactivity": "недавно",
                }
            )
        return result


@router.get("/api/datasets/{dataset_id}/preview/{image_path:path}")
async def get_dataset_preview_image(dataset_id: int, image_path: str):
    """Отдать изображение для превью датасета"""
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        full_image_path = os.path.join(dataset.path, image_path)

        if not os.path.exists(full_image_path):
            raise HTTPException(status_code=404, detail="Изображение не найдено")

        return FileResponse(full_image_path)


@router.post("/api/addDataset")
async def add_dataset(body: AddDatasetRequest):
    """добавить новый датасет по пути на диске"""

    try:
        container_path = resolve_container_path(body.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    print(container_path)

    # проверяем что путь существует
    if not os.path.exists(container_path):
        raise HTTPException(status_code=400, detail="указанный путь не существует")

    # считаем количество изображений
    total = 0
    for root, dirs, files in os.walk(container_path):
        total += len(
            [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

    with Session() as session:

        status = session.query(DatasetStatus).filter(DatasetStatus.id == 0).first()
        if not status:
            raise HTTPException(
                status_code=500, detail="статусы не инициализированы в БД"
            )

        dataset = Dataset(
            name=body.dataset_name,
            status_id=0,
            total_size=total,
            inwork_size=0,
            path=container_path,
            average_percent_success=None,
            current_model_architecture="yolo11n",
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)

        # Автоматически добавляем все изображения в БД
        images_dir = os.path.join(dataset.path, "images")
        if os.path.isdir(images_dir):
            # Получаем статус "unlabeled"
            unlabeled_status = (
                session.query(ImageStatus)
                .filter(ImageStatus.code == "unlabeled")
                .first()
            )
            default_status_id = unlabeled_status.id if unlabeled_status else 1

            # Сканируем папку с изображениями
            image_files = [
                f
                for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            # Создаем записи для каждого изображения
            for filename in image_files:
                existing = (
                    session.query(DatasetImage)
                    .filter(
                        DatasetImage.dataset_id == dataset.id,
                        DatasetImage.filename == filename,
                    )
                    .first()
                )

                if not existing:
                    new_image = DatasetImage(
                        dataset_id=dataset.id,
                        filename=filename,
                        status_id=default_status_id,
                    )
                    session.add(new_image)

            session.commit()

        return {
            "id": dataset.id,
            "name": dataset.name,
            "status_id": dataset.status_id,
            "status": {"id": status.id, "name": status.name},
            "total_size": dataset.total_size,
            "inwork_size": dataset.inwork_size,
            "path": dataset.path,
            "preview_image": None,
            "average_percent_success": dataset.average_percent_success,
            "current_model_architecture": dataset.current_model_architecture,
            "metric_precision": dataset.metric_precision,
            "metric_recall": dataset.metric_recall,
            "metric_f1": dataset.metric_f1,
            "metric_mean_iou": dataset.metric_mean_iou,
            "metrics_boxes_total": dataset.metrics_boxes_total,
            "metrics_images_total": dataset.metrics_images_total,
            "lastactivity": "недавно",
        }


@router.get("/api/models")
async def get_models():
    """список всех доступных архитектур"""
    return get_all_models()


@router.get("/api/datasets/{dataset_id}/model")
async def get_current_model(dataset_id: int):
    """текущая архитектура модели для датасета"""
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        return {"architecture": dataset.current_model_architecture}


@router.get("/api/datasets/{dataset_id}/next-batch")
def get_next_batch(dataset_id: int, limit: int = Query(100, ge=1, le=1000)):
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")

        active_batch = (
            session.query(PredictionBatch)
            .filter(
                PredictionBatch.dataset_id == dataset_id,
                PredictionBatch.status == "active",
            )
            .first()
        )
        if active_batch:
            raise HTTPException(
                status_code=409,
                detail="для этого датасета уже есть активный batch",
            )

        dataset_name = dataset.name
        dataset_path = dataset.path
        architecture = dataset.current_model_architecture

        active_model = (
            session.query(ModelVersion)
            .filter(
                ModelVersion.dataset_id == dataset_id,
                ModelVersion.is_active == True,
            )
            .order_by(ModelVersion.version.desc())
            .first()
        )
        model_version_id = active_model.id if active_model else None

    images_dir = os.path.join(dataset_path, "images")
    if not os.path.isdir(images_dir):
        raise HTTPException(
            status_code=404,
            detail=f"изображения датасета {dataset_name} не найдены",
        )

    annotator = get_annotator(dataset_id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    with Session() as session:
        image_filenames = select_batch_images_by_status(
            session=session,
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            limit=limit,
        )

    if not image_filenames:
        raise HTTPException(
            status_code=404,
            detail="в датасете нет изображений для batch",
        )

    images_payload = []
    snapshot_boxes: list[dict] = []

    for filename in image_filenames:
        image_path = os.path.join(dataset_path, "images", filename)
        boxes = annotator.predict(image_path)

        response_boxes = []
        for box in boxes:
            normalized_box = {
                "class_id": int(box["class_id"]),
                "class_name": box["class_name"],
                "confidence": float(box["confidence"]),
                "x1": int(box["x1"]),
                "y1": int(box["y1"]),
                "x2": int(box["x2"]),
                "y2": int(box["y2"]),
            }

            response_boxes.append(normalized_box)
            snapshot_boxes.append(
                {
                    "image_filename": filename,
                    "class_id": normalized_box["class_id"],
                    "confidence": normalized_box["confidence"],
                    "x1": normalized_box["x1"],
                    "y1": normalized_box["y1"],
                    "x2": normalized_box["x2"],
                    "y2": normalized_box["y2"],
                }
            )

        images_payload.append(
            {
                "filename": filename,
                "image_url": f"/api/datasets/{dataset_name}/images/{quote(filename)}",
                "boxes": response_boxes,
            }
        )

    completed_metrics = None
    should_complete_immediately = False

    with Session() as session:
        session.query(Dataset).filter(
            Dataset.id == dataset_id
        ).with_for_update().first()

        active_batch_race = (
            session.query(PredictionBatch)
            .filter(
                PredictionBatch.dataset_id == dataset_id,
                PredictionBatch.status == "active",
            )
            .first()
        )
        if active_batch_race:
            raise HTTPException(
                status_code=409,
                detail="для этого датасета уже есть активный batch",
            )

        config = (
            session.query(TrainingConfig)
            .filter(TrainingConfig.dataset_id == dataset_id)
            .first()
        )

        auto_accept_enabled = config.auto_accept_enabled if config else False
        auto_accept_threshold = (
            float(config.auto_accept_confidence_threshold)
            if config
            else 0.85
        )

        boxes_by_filename = {img["filename"]: img["boxes"] for img in images_payload}

        auto_accepted_set: set[str] = set()
        review_list: list[str] = []

        for filename in image_filenames:
            boxes = boxes_by_filename.get(filename, [])

            if (
                auto_accept_enabled
                and boxes
                and all(box["confidence"] >= auto_accept_threshold for box in boxes)
            ):
                auto_accepted_set.add(filename)
            else:
                review_list.append(filename)

        existing_images = {
            row.filename: row
            for row in session.query(DatasetImage)
            .filter(
                DatasetImage.dataset_id == dataset_id,
                DatasetImage.filename.in_(image_filenames),
            )
            .all()
        }

        ready_status = (
            session.query(ImageStatus)
            .filter(ImageStatus.code == "ready_for_training")
            .first()
        )
        ready_status_id = ready_status.id if ready_status else 3

        review_status = (
            session.query(ImageStatus)
            .filter(ImageStatus.code == "auto_labeled_pending_review")
            .first()
        )
        review_status_id = review_status.id if review_status else 4

        for filename in image_filenames:
            image_path = os.path.join(dataset_path, "images", filename)
            label_path = os.path.join(
                dataset_path,
                "labels",
                os.path.splitext(filename)[0] + ".txt",
            )

            row = existing_images.get(filename)
            target_status_id = (
                ready_status_id
                if filename in auto_accepted_set
                else review_status_id
            )

            if row:
                row.image_path = image_path
                row.label_path = label_path

                if row.status_id in (1, 4):
                    row.status_id = target_status_id
            else:
                session.add(
                    DatasetImage(
                        dataset_id=dataset_id,
                        filename=filename,
                        image_path=image_path,
                        label_path=label_path,
                        status_id=target_status_id,
                    )
                )

        session.flush()

        all_images_in_db = {
            row.filename: row
            for row in session.query(DatasetImage)
            .filter(
                DatasetImage.dataset_id == dataset_id,
                DatasetImage.filename.in_(image_filenames),
            )
            .all()
        }

        for filename in auto_accepted_set:
            save_auto_accepted_labels(
                dataset_path=dataset_path,
                filename=filename,
                boxes=boxes_by_filename.get(filename, []),
            )

        batch = PredictionBatch(
            dataset_id=dataset_id,
            model_version_id=model_version_id,
            architecture=architecture,
            status="active",
            image_filenames_json=json.dumps(image_filenames),
            auto_accepted_filenames_json=json.dumps(sorted(auto_accepted_set)),
            review_filenames_json=json.dumps(review_list),
        )
        session.add(batch)
        session.flush()

        for box in snapshot_boxes:
            img_row = all_images_in_db.get(box["image_filename"])

            session.add(
                PredictionBox(
                    batch_id=batch.id,
                    dataset_image_id=img_row.id if img_row else None,
                    image_filename=box["image_filename"],
                    class_id=box["class_id"],
                    confidence=box["confidence"],
                    x1=box["x1"],
                    y1=box["y1"],
                    x2=box["x2"],
                    y2=box["y2"],
                )
            )

        batch_id = batch.id
        should_complete_immediately = not review_list
        session.commit()

    if should_complete_immediately:
        completed_metrics = complete_batch_and_update_dataset_metrics(
            dataset_id=dataset_id,
            batch_id=batch_id,
        )

    review_images_payload = [
        img for img in images_payload if img["filename"] in set(review_list)
    ]

    return {
        "batch_id": batch_id,
        "dataset_id": dataset_id,
        "images": review_images_payload,
        "auto_accepted_count": len(auto_accepted_set),
        "review_count": len(review_list),
        "total_count": len(image_filenames),
        "batch_status": "completed" if completed_metrics else "active",
        "completed_metrics": completed_metrics,
    }

@router.post("/api/datasets/{dataset_id}/batches/{batch_id}/complete")
def complete_prediction_batch(dataset_id: int, batch_id: int):
    metrics = complete_batch_and_update_dataset_metrics(
        dataset_id=dataset_id,
        batch_id=batch_id,
    )

    return {
        "status": "completed",
        "batch_id": batch_id,
        "dataset_id": dataset_id,
        "metrics": metrics,
    }


@router.post("/api/datasets/{dataset_id}/model")
async def change_model(dataset_id: int, body: ChangeModelRequest):
    """запланировать смену архитектуры модели для датасета через retrain в очереди"""

    model_info = get_model_by_id(body.architecture)
    if not model_info:
        raise HTTPException(
            status_code=400, detail=f"неизвестная архитектура: {body.architecture}"
        )

    job_id = None
    with Session() as session:
        dataset = (
            session.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .with_for_update()
            .first()
        )
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")

        active_job = (
            session.query(TrainingJob)
            .filter(
                TrainingJob.dataset_id == dataset.id,
                TrainingJob.status.in_(["queued", "running"]),
            )
            .first()
        )
        if active_job:
            raise HTTPException(
                status_code=409, detail="для датасета уже выполняется training job"
            )

        if (
            body.architecture == dataset.current_model_architecture
            and dataset.pending_model_architecture is None
        ):
            raise HTTPException(
                status_code=409, detail="архитектура уже активна для этого датасета"
            )

        labeled_statuses = (
            session.query(ImageStatus)
            .filter(
                ImageStatus.code.in_(["labeled", "finalized", "ready_for_training"])
            )
            .all()
        )
        status_ids = [s.id for s in labeled_statuses]

        has_labels = (
            session.query(DatasetImage)
            .filter(
                DatasetImage.dataset_id == dataset.id,
                DatasetImage.status_id.in_(status_ids),
            )
            .first()
            is not None
        )

        if not has_labels:
            dataset.current_model_architecture = body.architecture
            dataset.pending_model_architecture = None
            session.commit()
            invalidate_annotator(dataset_id)
            return {"status": "ok", "architecture": body.architecture}

        dataset.pending_model_architecture = body.architecture

        job = TrainingJob(
            dataset_id=dataset.id, status="queued", job_type="change_model_retrain"
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    try:
        submit_training_job(job_id)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="не удалось поставить retrain смены архитектуры в очередь",
        )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "target_architecture": body.architecture,
        },
    )

@router.get("/api/datasets/{dataset_id}/hyperparams")
async def get_hyperparams(dataset_id: int):
    """гиперпараметры для текущей модели датасета"""
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        config = (
            session.query(TrainingConfig)
            .filter(TrainingConfig.dataset_id == dataset_id)
            .first()
        )
        if not config:
            # Если конфига нет, сразу инициализируем с автодетектом железа
            auto_dev = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
            config = TrainingConfig(dataset_id=dataset_id, device=auto_dev)
            session.add(config)
            session.commit()
            session.refresh(config)
        return {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "imgsz": config.imgsz,
            "optimizer": config.optimizer,
            "device": config.device,
            "augmentation_enabled": config.augmentation_enabled,
            "augmentation_threshold": config.augmentation_threshold,
            "auto_accept_enabled": config.auto_accept_enabled,
            "auto_accept_confidence_threshold": config.auto_accept_confidence_threshold,
            "incremental_training_enabled": config.incremental_training_enabled
        }


@router.put("/api/datasets/{dataset_id}/hyperparams")
async def update_hyperparams(dataset_id: int, body: TrainingConfigRequest):
    """обновить гиперпараметры для текущей модели датасета"""
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        config = (
            session.query(TrainingConfig)
            .filter(TrainingConfig.dataset_id == dataset_id)
            .first()
        )
        if not config:
            auto_dev = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
            config = TrainingConfig(dataset_id=dataset_id, device=auto_dev)
            session.add(config)

        device_changed = False

        if body.epochs is not None:
            config.epochs = body.epochs
        if body.batch_size is not None:
            config.batch_size = body.batch_size
        if body.learning_rate is not None:
            config.learning_rate = body.learning_rate
        if body.imgsz is not None:
            config.imgsz = body.imgsz
        if body.optimizer is not None:
            config.optimizer = body.optimizer

        if body.augmentation_enabled is not None:
            config.augmentation_enabled = body.augmentation_enabled
        if body.augmentation_threshold is not None:
            config.augmentation_threshold = body.augmentation_threshold

        if body.auto_accept_enabled is not None:
            config.auto_accept_enabled = body.auto_accept_enabled
        if body.auto_accept_confidence_threshold is not None:
            config.auto_accept_confidence_threshold = body.auto_accept_confidence_threshold

        if body.incremental_training_enabled is not None:
            config.incremental_training_enabled = body.incremental_training_enabled

        # Проверяем, изменился ли девайс, чтобы знать нужно ли сбрасывать кэш
        if body.device is not None and config.device != body.device:
            config.device = body.device
            device_changed = True

        session.commit()

        # Если девайс поменяли, выкидываем старую модель из оперативки.
        # Следующий вызов get_annotator создаст её заново уже на новом железе.
        if device_changed:
            invalidate_annotator(dataset_id)

        return {"status": "ok"}


@router.get("/api/datasets/{dataset_id}/metrics")
async def get_metrics(dataset_id: int):
    """Эндпоинт для получения метрик всех версий модели для указанного датасета (precision, recall, f1, mAP и т.д.)."""
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")

        # загружаем все версии модели для датасета, отсортированные по возрастанию версии
        model_versions = (
            session.execute(
                select(ModelVersion)
                .where(ModelVersion.dataset_id == dataset_id)
                .order_by(ModelVersion.version.asc())
            )
            .scalars()
            .all()
        )

        # формируем список метрик каждой версии, десериализуя confusion_matrix из JSON
        return [
            {
                "version": v.version,
                "architecture": v.architecture,
                "epochs": v.epochs,
                "is_active": v.is_active,
                "created_at": v.created_at.isoformat(),
                "precision": v.precision,
                "recall": v.recall,
                "f1": v.f1,
                "map50": v.map50,
                "map50_95": v.map50_95,
                "mean_iou": v.mean_iou,
                "confusion_matrix": (
                    json.loads(v.confusion_matrix_json)
                    if v.confusion_matrix_json
                    else None
                ),
                "mlflow_run_id": v.mlflow_run_id,
            }
            for v in model_versions
        ]


@router.get("/api/datasets/{dataset_id}/metrics/latest")
async def get_curr_metrics(dataset_id: int):
    """Эндпоинт для получения метрик текущей активной версии модели для указанного датасета."""
    with Session() as session:

        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")

        # ищем единственную активную версию модели (is_active=True) для датасета
        model_vers_lts = session.execute(
            select(ModelVersion)
            .where(ModelVersion.dataset_id == dataset_id)
            .where(ModelVersion.is_active == True)
        ).scalar_one_or_none()

        if not model_vers_lts:
            raise HTTPException(status_code=404, detail="Нет Активной модели")

        v = model_vers_lts
        return {
            "version": v.version,
            "architecture": v.architecture,
            "epochs": v.epochs,
            "is_active": v.is_active,
            "created_at": v.created_at.isoformat(),
            "precision": v.precision,
            "recall": v.recall,
            "f1": v.f1,
            "map50": v.map50,
            "map50_95": v.map50_95,
            "mean_iou": v.mean_iou,
            "confusion_matrix": (
                json.loads(v.confusion_matrix_json) if v.confusion_matrix_json else None
            ),
            "mlflow_run_id": v.mlflow_run_id,
        }


@router.get("/api/datasets/{dataset_id}/augmentation")
async def get_augmentation(dataset_id: int):
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        config = (
            session.query(TrainingConfig)
            .filter(TrainingConfig.dataset_id == dataset_id)
            .first()
        )
        if not config:
            config = TrainingConfig(dataset_id=dataset_id)
            session.add(config)
            session.commit()
            session.refresh(config)
        return {
            "augmentation_enabled": config.augmentation_enabled,
            "augmentation_threshold": config.augmentation_threshold,
        }


@router.put("/api/datasets/{dataset_id}/augmentation")
async def update_augmentation(dataset_id: int, body: AugmentationConfigRequest):
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        config = (
            session.query(TrainingConfig)
            .filter(TrainingConfig.dataset_id == dataset_id)
            .first()
        )
        if not config:
            config = TrainingConfig(dataset_id=dataset_id)
            session.add(config)
            session.commit()
            session.refresh(config)
        if body.augmentation_enabled is not None:
            config.augmentation_enabled = body.augmentation_enabled
        if body.augmentation_threshold is not None:
            config.augmentation_threshold = body.augmentation_threshold

        session.commit()
        return {"status": "ok"}


# New endpoints for work page


class BoundingBoxData(BaseModel):
    id: Optional[int] = None
    class_id: int
    label: str
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


@router.get("/api/datasets/{dataset_id:int}/images")
async def get_dataset_images(dataset_id: int, status: Optional[str] = None):
    """Get images from DB filtered by status"""
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        # Query images from DB
        query = session.query(DatasetImage).filter(
            DatasetImage.dataset_id == dataset_id
        )

        # Filter by status if provided
        if status:
            status_obj = (
                session.query(ImageStatus).filter(ImageStatus.code == status).first()
            )
            if status_obj:
                query = query.filter(DatasetImage.status_id == status_obj.id)

        db_images = query.order_by(DatasetImage.filename).all()

        # Get status code for each image
        result = []
        for img in db_images:
            img_status = (
                session.query(ImageStatus)
                .filter(ImageStatus.id == img.status_id)
                .first()
            )
            status_code = img_status.code if img_status else "unlabeled"

            result.append(
                {
                    "id": img.id,
                    "filename": img.filename,
                    "url": f"/api/datasets/{dataset_id}/image/{quote(img.filename)}",
                    "preview_url": f"/api/datasets/{dataset_id}/image/{quote(img.filename)}",
                    "status_id": img.status_id,
                    "status_code": status_code,
                }
            )

        return result


@router.post("/api/datasets/{dataset_id:int}/create-image")
async def create_image_record(dataset_id: int, body: dict):
    """Create image record in DB"""
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        filename = body.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="filename is required")

        # Check if already exists
        existing = (
            session.query(DatasetImage)
            .filter(
                DatasetImage.dataset_id == dataset_id, DatasetImage.filename == filename
            )
            .first()
        )

        if existing:
            return {
                "id": existing.id,
                "filename": existing.filename,
                "status_id": existing.status_id,
            }

        # Get default status (unlabeled)
        unlabeled_status = (
            session.query(ImageStatus).filter(ImageStatus.code == "unlabeled").first()
        )
        default_status_id = unlabeled_status.id if unlabeled_status else 1

        new_image = DatasetImage(
            dataset_id=dataset_id, filename=filename, status_id=default_status_id
        )
        session.add(new_image)
        session.commit()
        session.refresh(new_image)

        return {
            "id": new_image.id,
            "filename": new_image.filename,
            "status_id": new_image.status_id,
        }


@router.get("/api/datasets/{dataset_id:int}/image/{filename:path}")
async def get_dataset_image(dataset_id: int, filename: str):
    """Serve an image from dataset"""
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        image_path = os.path.join(dataset.path, "images", filename)

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Изображение не найдено")

        return FileResponse(image_path)


@router.get("/api/images/{image_id:int}/detections")
async def get_image_detections(image_id: int):
    """Get bounding boxes for an image"""
    with Session() as session:
        image = session.query(DatasetImage).filter(DatasetImage.id == image_id).first()
        if not image:
            # Return empty array if image not in DB yet
            return []

        # Get bounding boxes from prediction_box table
        boxes = (
            session.query(PredictionBox)
            .filter(
                PredictionBox.dataset_image_id == image_id,
                PredictionBox.batch_id.is_(None),
            ).all()
        )

        result = []
        for box in boxes:
            # Find class by class_id and dataset_id
            bbox_class = (
                session.query(BoundingBoxClass)
                .filter(
                    BoundingBoxClass.class_id == box.class_id,
                    BoundingBoxClass.dataset_id == image.dataset_id,
                )
                .first()
            )

            result.append(
                {
                    "id": box.id,
                    "class_id": box.class_id,
                    "label": bbox_class.name if bbox_class else "Unknown",
                    "cls": (
                        bbox_class.color
                        if (bbox_class and bbox_class.color)
                        else "blue"
                    ),
                    "conf": box.confidence,
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                }
            )

        return result


@router.post("/api/images/{image_id:int}/detections")
async def save_image_detections(
    image_id: int, detections: list[BoundingBoxData], is_auto: bool = False
):
    """Save bounding boxes for an image"""
    with Session() as session:
        image = session.query(DatasetImage).filter(DatasetImage.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Изображение не найдено")

        dataset = session.query(Dataset).filter(Dataset.id == image.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        filename = image.filename
        image_path = os.path.join(dataset.path, "images", filename)
        label_path = os.path.join(
            dataset.path,
            "labels",
            os.path.splitext(filename)[0] + ".txt",
        )

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Файл изображения не найден")

        session.query(PredictionBox).filter(
            PredictionBox.dataset_image_id == image_id,
            PredictionBox.batch_id.is_(None),
        ).delete(synchronize_session=False)

        label_boxes = []

        for det in detections:
            existing_class = (
                session.query(BoundingBoxClass)
                .filter(
                    BoundingBoxClass.dataset_id == image.dataset_id,
                    BoundingBoxClass.class_id == det.class_id,
                )
                .first()
            )

            if not existing_class and hasattr(det, "label") and det.label:
                new_class = BoundingBoxClass(
                    dataset_id=image.dataset_id,
                    class_id=det.class_id,
                    name=det.label,
                    color="#3B82F6",
                )
                session.add(new_class)
                session.flush()

            box = PredictionBox(
                dataset_image_id=image_id,
                image_filename=filename,
                class_id=det.class_id,
                confidence=det.conf,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
            )
            session.add(box)

            label_boxes.append(
                {
                    "class_id": int(det.class_id),
                    "confidence": float(det.conf),
                    "x1": float(det.x1),
                    "y1": float(det.y1),
                    "x2": float(det.x2),
                    "y2": float(det.y2),
                }
            )

        save_auto_accepted_labels(
            dataset_path=dataset.path,
            filename=filename,
            boxes=label_boxes,
        )

        image.image_path = image_path
        image.label_path = label_path

        if is_auto:
            auto_status = (
                session.query(ImageStatus)
                .filter(ImageStatus.code == "auto_labeled_pending_review")
                .first()
            )
            if auto_status:
                image.status_id = auto_status.id
        else:
            labeled_status = (
                session.query(ImageStatus).filter(ImageStatus.code == "labeled").first()
            )
            if labeled_status:
                image.status_id = labeled_status.id

        _sync_dataset_status(session, image.dataset_id)
        session.commit()
        return {"success": True, "count": len(detections), "label_path": label_path}
    

@router.post("/api/images/{image_id:int}/accept")
async def accept_image(image_id: int):
    """Принять auto-labeled image и убедиться, что GT label-файл существует."""
    with Session() as session:
        image = session.query(DatasetImage).filter(DatasetImage.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Изображение не найдено")

        dataset = session.query(Dataset).filter(Dataset.id == image.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        boxes = (
            session.query(PredictionBox)
            .filter(
                PredictionBox.dataset_image_id == image_id,
                PredictionBox.batch_id.is_(None),
            )
            .all()
        )

        label_boxes = [
            {
                "class_id": box.class_id,
                "confidence": box.confidence,
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2,
            }
            for box in boxes
        ]

        save_auto_accepted_labels(
            dataset_path=dataset.path,
            filename=image.filename,
            boxes=label_boxes,
        )

        image.image_path = os.path.join(dataset.path, "images", image.filename)
        image.label_path = os.path.join(
            dataset.path,
            "labels",
            os.path.splitext(image.filename)[0] + ".txt",
        )

        ready_status = (
            session.query(ImageStatus)
            .filter(ImageStatus.code == "ready_for_training")
            .first()
        )
        if ready_status:
            image.status_id = ready_status.id

        _sync_dataset_status(session, image.dataset_id)
        session.commit()

        return {
            "success": True,
            "label_path": image.label_path,
        }


@router.post("/api/images/{image_id:int}/reject")
async def reject_image(image_id: int):
    """Отклонить auto-labeled image, вернуть в unlabeled и удалить временную разметку."""
    with Session() as session:
        image = session.query(DatasetImage).filter(DatasetImage.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Изображение не найдено")

        dataset = session.query(Dataset).filter(Dataset.id == image.dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        unlabeled_status = (
            session.query(ImageStatus)
            .filter(ImageStatus.code == "unlabeled")
            .first()
        )
        if unlabeled_status:
            image.status_id = unlabeled_status.id

        session.query(PredictionBox).filter(
            PredictionBox.dataset_image_id == image_id,
            PredictionBox.batch_id.is_(None),
        ).delete(synchronize_session=False)

        label_path = os.path.join(
            dataset.path,
            "labels",
            os.path.splitext(image.filename)[0] + ".txt",
        )
        if os.path.exists(label_path):
            os.remove(label_path)

        image.label_path = None

        _sync_dataset_status(session, image.dataset_id)
        session.commit()

        return {"success": True}

@router.get("/api/datasets/{dataset_id:int}/classes")
async def get_dataset_classes(dataset_id: int):
    """Get all classes for a dataset"""
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        classes = (
            session.query(BoundingBoxClass)
            .filter(BoundingBoxClass.dataset_id == dataset_id)
            .order_by(BoundingBoxClass.class_id)
            .all()
        )
        return [
            {
                "id": cls.id,
                "class_id": cls.class_id,
                "name": cls.name,
                "color": cls.color or "#3B82F6",
            }
            for cls in classes
        ]

class AddClassRequest(BaseModel):
    class_id: int
    name: str
    color: Optional[str] = "#3B82F6"


@router.post("/api/datasets/{dataset_id:int}/classes")
async def add_dataset_class(dataset_id: int, body: AddClassRequest):
    """Add a new class to dataset"""
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        # Check if class already exists
        existing = (
            session.query(BoundingBoxClass)
            .filter(
                BoundingBoxClass.dataset_id == dataset_id,
                BoundingBoxClass.class_id == body.class_id,
            )
            .first()
        )

        if existing:
            raise HTTPException(
                status_code=409, detail="Класс с таким class_id уже существует"
            )

        new_class = BoundingBoxClass(
            dataset_id=dataset_id,
            class_id=body.class_id,
            name=body.name,
            color=body.color,
        )
        session.add(new_class)
        session.commit()
        session.refresh(new_class)

        return {
            "id": new_class.id,
            "class_id": new_class.class_id,
            "name": new_class.name,
            "color": new_class.color,
        }


@router.delete("/api/datasets/{dataset_id:int}/classes/{class_id:int}")
async def delete_dataset_class(dataset_id: int, class_id: int):
    """Delete a class from dataset"""
    with Session() as session:
        cls = (
            session.query(BoundingBoxClass)
            .filter(BoundingBoxClass.id == class_id)
            .first()
        )
        if not cls or cls.dataset_id != dataset_id:
            raise HTTPException(status_code=404, detail="Класс не найден")

        session.delete(cls)
        session.commit()
        return {"success": True}


@router.post("/api/datasets/{dataset_id:int}/prepare-training")
async def prepare_training_data(dataset_id: int):
    """
    Проверяет готовность данных к обучению.

    В новой архитектуре GT-разметка хранится в dataset.path/labels/*.txt.
    Этот endpoint больше не конвертирует PredictionBox в labels, чтобы не
    перезаписывать настоящие GT label-файлы временными prediction snapshot.
    """
    from PIL import Image
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        images_dir = os.path.join(dataset.path, "images")
        labels_dir = os.path.join(dataset.path, "labels")

        if not os.path.isdir(images_dir):
            raise HTTPException(
                status_code=400,
                detail="Папка images не найдена для датасета",
            )

        os.makedirs(labels_dir, exist_ok=True)

        trainable_statuses = (
            session.query(ImageStatus)
            .filter(
                ImageStatus.code.in_(
                    ["labeled", "finalized", "ready_for_training"]
                )
            )
            .all()
        )
        status_ids = [status.id for status in trainable_statuses]

        images = (
            session.query(DatasetImage)
            .filter(
                DatasetImage.dataset_id == dataset_id,
                DatasetImage.status_id.in_(status_ids),
            )
            .all()
        )

        if not images:
            raise HTTPException(
                status_code=400,
                detail="Нет изображений со статусом, подходящим для обучения",
            )

        ready_count = 0
        missing_labels = []
        missing_images = []

        for image in images:
            image_path = os.path.join(dataset.path, "images", image.filename)
            label_path = os.path.join(
                dataset.path,
                "labels",
                os.path.splitext(image.filename)[0] + ".txt",
            )

            image.image_path = image_path
            image.label_path = label_path

            if not os.path.exists(image_path):
                missing_images.append(image.filename)
                continue

            if not os.path.exists(label_path):
                missing_labels.append(image.filename)
                continue

            ready_count += 1

        session.commit()

        if ready_count == 0:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Нет готовых изображений с GT label-файлами",
                    "missing_images": missing_images,
                    "missing_labels": missing_labels,
                },
            )

        return {
            "success": True,
            "total_trainable_images": len(images),
            "ready_for_training": ready_count,
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "labels_dir": labels_dir,
        }

@router.post("/api/datasets/{dataset_id:int}/train")
async def start_training(dataset_id: int):
    """Start training on labeled data"""
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        # Check if training is already running
        active_job = (
            session.query(TrainingJob)
            .filter(
                TrainingJob.dataset_id == dataset_id,
                TrainingJob.status.in_(["queued", "running"]),
            )
            .first()
        )

        if active_job:
            raise HTTPException(status_code=409, detail="Обучение уже запущено")

        # Create training job
        job = TrainingJob(dataset_id=dataset_id, status="queued", job_type="train")
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    try:
        submit_training_job(job_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Не удалось запустить обучение: {str(e)}"
        )

    return {"job_id": job_id, "status": "queued"}


@router.get("/api/datasets/{dataset_id:int}/stats")
async def get_dataset_stats(dataset_id: int):
    """Статистика по датасету: счётчики изображений, уверенность, распределение классов, метрики модели."""
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Датасет не найден")

        all_images = session.query(DatasetImage).filter(
            DatasetImage.dataset_id == dataset_id
        ).all()

        statuses = session.query(ImageStatus).all()
        unlabeled_ids = {s.id for s in statuses if s.code == "unlabeled"}
        done_ids = {s.id for s in statuses if s.code in ("labeled", "ready_for_training", "finalized")}
        pending_ids = {s.id for s in statuses if s.code == "auto_labeled_pending_review"}

        total = len(all_images)
        labeled = sum(1 for img in all_images if img.status_id in done_ids)
        pending = sum(1 for img in all_images if img.status_id in pending_ids)
        unlabeled = sum(1 for img in all_images if img.status_id in unlabeled_ids)

        # Средняя уверенность по боксам датасета
        image_ids = [img.id for img in all_images]
        boxes = (
            session.query(PredictionBox)
            .filter(PredictionBox.dataset_image_id.in_(image_ids))
            .all()
        ) if image_ids else []

        avg_conf = (
            round(sum(b.confidence for b in boxes) / len(boxes) * 100, 1)
            if boxes else None
        )

        # Распределение по классам
        class_counts: dict[int, int] = {}
        for box in boxes:
            class_counts[box.class_id] = class_counts.get(box.class_id, 0) + 1

        classes = session.query(BoundingBoxClass).filter(
            BoundingBoxClass.dataset_id == dataset_id
        ).all()
        class_name_map = {c.class_id: c.name for c in classes}

        total_boxes = sum(class_counts.values()) or 1
        class_distribution = sorted(
            [
                {
                    "class_id": cid,
                    "name": class_name_map.get(cid, f"Класс {cid}"),
                    "count": cnt,
                    "percent": round(cnt / total_boxes * 100, 1),
                }
                for cid, cnt in class_counts.items()
            ],
            key=lambda x: x["count"],
            reverse=True,
        )

        # История метрик по версиям модели
        model_versions = (
            session.query(ModelVersion)
            .filter(ModelVersion.dataset_id == dataset_id)
            .order_by(ModelVersion.version.asc())
            .all()
        )
        metrics_history = [
            {
                "version": v.version,
                "precision": v.precision,
                "recall": v.recall,
                "f1": v.f1,
                "map50": v.map50,
            }
            for v in model_versions
        ]

        # История активности: последние 10 батчей авторазметки
        batches = (
            session.query(PredictionBatch)
            .filter(PredictionBatch.dataset_id == dataset_id)
            .order_by(PredictionBatch.created_at.desc())
            .limit(10)
            .all()
        )

        activity = []
        for batch in batches:
            # Количество изображений в батче
            try:
                filenames = json.loads(batch.image_filenames_json) if batch.image_filenames_json else []
                images_count = len(filenames)
            except Exception:
                images_count = 0

            # Средняя уверенность боксов батча
            batch_boxes = (
                session.query(PredictionBox)
                .filter(PredictionBox.batch_id == batch.id)
                .all()
            )
            batch_avg_conf = (
                round(sum(b.confidence for b in batch_boxes) / len(batch_boxes) * 100, 1)
                if batch_boxes else None
            )

            activity.append({
                "batch_id": batch.id,
                "type": "auto",
                "images_count": images_count,
                "avg_confidence": batch_avg_conf,
                "architecture": batch.architecture,
                "created_at": batch.created_at.isoformat(),
            })

        return {
            "dataset_name": dataset.name,
            "total_images": total,
            "labeled": labeled,
            "pending_review": pending,
            "unlabeled": unlabeled,
            "avg_confidence": avg_conf,
            "total_boxes": sum(class_counts.values()),
            "class_distribution": class_distribution,
            "metrics_history": metrics_history,
            "activity": activity,
        }

@router.get("/api/system/devices")
async def get_available_devices():
    """определяет что доступно из девайсов"""
    devices = [{"id": "cpu", "name": "CPU (Процессор)"}]
    auto_device = "cpu"

    if torch.cuda.is_available():
        devices.append({"id": "cuda", "name": "CUDA (NVIDIA GPU)"})
        # Можно даже вытащить имя видяхи: torch.cuda.get_device_name(0)
        auto_device = "cuda"
    elif torch.backends.mps.is_available():
        devices.append({"id": "mps", "name": "MPS (Apple Silicon)"})
        auto_device = "mps"

    return {"devices": devices, "auto_detect": auto_device}

@router.get("/api/datasets/{dataset_id}/training-status")
async def get_training_status(dataset_id: int):
    with Session() as session:
        job = (
            session.query(TrainingJob)
            .filter(TrainingJob.dataset_id == dataset_id)
            .order_by(TrainingJob.created_at.desc())
            .first()
        )
        if not job:
            return {"status": "idle"}

        return {
            "job_id": job.id,
            "job_type": job.job_type,
            "status": job.status,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "model_version_id": job.model_version_id,
        }