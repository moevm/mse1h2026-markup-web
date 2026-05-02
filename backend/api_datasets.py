from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from activate import Session
from db import (
    DatasetImage,
    Dataset,
    DatasetStatus,
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
from helper import get_annotator, invalidate_annotator
import json
from urllib.parse import quote

DATASETS_ROOT_HOST = "C:/"
DATASETS_ROOT_CONTAINER = "/host_c"

def resolve_container_path(user_path: str) -> str:
    user_path = user_path.replace("\\", "/")
    host_root = DATASETS_ROOT_HOST.replace("\\", "/").rstrip("/")
    
    if not user_path.lower().startswith(host_root.lower()):
        raise ValueError(f"Путь должен находиться внутри {host_root}")
    
    relative = user_path[len(host_root):].lstrip("/")
    return os.path.join(DATASETS_ROOT_CONTAINER, relative)


def dataset_has_labels(dataset_path: str) -> bool:
    labels_dir = os.path.join(dataset_path, "labels", "train")
    if not os.path.isdir(labels_dir):
        return False

    for _, _, files in os.walk(labels_dir):
        for filename in files:
            if filename.lower().endswith(".txt"):
                return True
    return False


def list_dataset_images(dataset_path: str) -> list[str]:
    if not os.path.isdir(dataset_path):
        return []

    return sorted(
        [
            filename
            for filename in os.listdir(dataset_path)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.isfile(os.path.join(dataset_path, filename))
        ]
    )


def select_batch_images_by_status(session, dataset_id: int, dataset_path: str, limit: int) -> list[str]:
    image_filenames = list_dataset_images(dataset_path)
    if not image_filenames:
        return []

    existing_rows = (
        session.query(DatasetImage)
        .filter(
            DatasetImage.dataset_id == dataset_id,
            DatasetImage.filename.in_(image_filenames)
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


class AddDatasetRequest(BaseModel):
    dataset_name: str
    path: str


class ChangeModelRequest(BaseModel):
    architecture: str


class TrainingConfigRequest(BaseModel):
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    imgsz: Optional[int] = None
    optimizer: Optional[str] = None


class AugmentationConfigRequest(BaseModel):
    augmentation_enabled: Optional[bool] = None
    augmentation_threshold: Optional[float] = None

class ClassItem(BaseModel):
    class_id: int
    name: str
    color: Optional[str] = None

class ClassesRequest(BaseModel):
    classes: list[ClassItem]

@router.get("/api/datasets/{dataset_id}/classes")
async def get_dataset_classes(dataset_id: int):
    with Session() as session:
        classes = (
            session.query(BoundingBoxClass)
            .filter(BoundingBoxClass.dataset_id == dataset_id)
            .order_by(BoundingBoxClass.class_id)
            .all()
        )
        return [
            {"class_id": c.class_id, "name": c.name, "color": c.color}
            for c in classes
        ]

@router.post("/api/datasets/{dataset_id}/classes")
async def update_dataset_classes(dataset_id: int, body: ClassesRequest):
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        
        session.query(BoundingBoxClass).filter(
            BoundingBoxClass.dataset_id == dataset_id
        ).delete()
        
        for item in body.classes:
            session.add(
                BoundingBoxClass(
                    dataset_id=dataset_id,
                    class_id=item.class_id,
                    name=item.name,
                    color=item.color
                )
            )
        session.commit()
    return {"status": "ok"}


@router.get("/api/getDatasets")
async def get_datasets():
    '''список всех датасетов в формате для фронта'''
    with Session() as session:
        datasets = session.query(Dataset).all()
        result = []
        for ds in datasets:
            status = session.query(DatasetStatus).filter(DatasetStatus.id == ds.status_id).first()
            result.append({
                "id": ds.id,
                "name": ds.name,
                "status": {
                    "id": status.id,
                    "name": status.name
                },
                "total_size": ds.total_size,
                "inwork_size": ds.inwork_size,
                "path": ds.path,
                "average_percent_success": ds.average_percent_success,
                "current_model_architecture": ds.current_model_architecture,
                "metric_precision": ds.metric_precision,
                "metric_recall": ds.metric_recall,
                "metric_f1": ds.metric_f1,
                "metric_mean_iou": ds.metric_mean_iou,
                "metrics_boxes_total": ds.metrics_boxes_total,
                "metrics_images_total": ds.metrics_images_total,
            })
        return result


@router.post("/api/addDataset")
async def add_dataset(body: AddDatasetRequest):
    '''добавить новый датасет по пути на диске'''

    try:
        container_path = resolve_container_path(body.path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


    # проверяем что путь существует
    if not os.path.exists(container_path):
        raise HTTPException(status_code=400, detail="указанный путь не существует")

    # считаем количество изображений
    total = 0
    for root, dirs, files in os.walk(container_path):
        total += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    with Session() as session:
        
        status = session.query(DatasetStatus).filter(DatasetStatus.id == 0).first()
        if not status:
            raise HTTPException(status_code=500, detail="статусы не инициализированы в БД")

        dataset = Dataset(
            name=body.dataset_name,
            status_id=0,       
            total_size=total,
            inwork_size=0,
            path=container_path,
            average_percent_success=None
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)

        return {
            "id": dataset.id,
            "name": dataset.name,
            "status": {"id": 0, "name": "Just Load"},
            "total_size": dataset.total_size,
            "inwork_size": 0,
            "path": dataset.path,
            "average_percent_success": None,
            "current_model_architecture": dataset.current_model_architecture,
            "metric_precision": dataset.metric_precision,
            "metric_recall": dataset.metric_recall,
            "metric_f1": dataset.metric_f1,
            "metric_mean_iou": dataset.metric_mean_iou,
            "metrics_boxes_total": dataset.metrics_boxes_total,
            "metrics_images_total": dataset.metrics_images_total,
        }

@router.get("/api/models")
async def get_models():
    '''список всех доступных архитектур'''
    return get_all_models()


@router.get("/api/datasets/{dataset_id}/model")
async def get_current_model(dataset_id: int):
    '''текущая архитектура модели для датасета'''
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
                PredictionBatch.status == "active"
            )
            .first()
        )
        if active_batch:
            raise HTTPException(
                status_code=409,
                detail="для этого датасета уже есть активный batch"
            )

        dataset_name = dataset.name
        dataset_path = dataset.path
        architecture = dataset.current_model_architecture

        active_model = (
            session.query(ModelVersion)
            .filter(
                ModelVersion.dataset_id == dataset_id,
                ModelVersion.is_active == True
            )
            .order_by(ModelVersion.version.desc())
            .first()
        )
        model_version_id = active_model.id if active_model else None

    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404, detail=f"изображения датасета {dataset_name} не найдены"
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
        raise HTTPException(status_code=404, detail="в датасете нет изображений для batch")

    images_payload = []
    snapshot_boxes: list[dict] = []
    for filename in image_filenames:
        image_path = os.path.join(dataset_path, filename)
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

    with Session() as session:
        session.query(Dataset).filter(Dataset.id == dataset_id).with_for_update().first()
        active_batch_race = (
            session.query(PredictionBatch)
            .filter(
                PredictionBatch.dataset_id == dataset_id,
                PredictionBatch.status == "active"
            )
            .first()
        )
        if active_batch_race:
            raise HTTPException(
                status_code=409,
                detail="для этого датасета уже есть активный batch"
            )

        existing_images = {
            row.filename: row
            for row in session.query(DatasetImage).filter(
                DatasetImage.dataset_id == dataset_id,
                DatasetImage.filename.in_(image_filenames)
            ).all()
        }

        for filename in image_filenames:
            image_path = os.path.join(dataset_path, filename)
            label_path = os.path.join(
                dataset_path,
                "labels",
                "train",
                os.path.splitext(filename)[0] + ".txt"
            )
            row = existing_images.get(filename)
            if row:
                row.image_path = image_path
                row.label_path = label_path
                if row.status_id == 1:
                    row.status_id = 4
            else:
                session.add(
                    DatasetImage(
                        dataset_id=dataset_id,
                        filename=filename,
                        image_path=image_path,
                        label_path=label_path,
                        status_id=4,
                    )
                )

        batch = PredictionBatch(
            dataset_id=dataset_id,
            model_version_id=model_version_id,
            architecture=architecture,
            status="active",
            image_filenames_json=json.dumps(image_filenames),
        )
        session.add(batch)
        session.flush()

        for box in snapshot_boxes:
            session.add(
                PredictionBox(
                    batch_id=batch.id,
                    image_filename=box["image_filename"],
                    class_id=box["class_id"],
                    confidence=box["confidence"],
                    x1=box["x1"],
                    y1=box["y1"],
                    x2=box["x2"],
                    y2=box["y2"],
                )
            )

        session.commit()
        batch_id = batch.id

    return {
        "batch_id": batch_id,
        "dataset_id": dataset_id,
        "images": images_payload,
    }


@router.post("/api/datasets/{dataset_id}/model")
async def change_model(dataset_id: int, body: ChangeModelRequest):
    '''запланировать смену архитектуры модели для датасета через retrain в очереди'''

    # проверяем что такая модель существует в реестре
    model_info = get_model_by_id(body.architecture)
    if not model_info:
        raise HTTPException(status_code=400, detail=f"неизвестная архитектура: {body.architecture}")

    job_id = None
    has_labels = False
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
                TrainingJob.status.in_(["queued", "running"])
            )
            .first()
        )
        if active_job:
            raise HTTPException(
                status_code=409,
                detail="для датасета уже выполняется training job"
            )

        if (
            body.architecture == dataset.current_model_architecture
            and dataset.pending_model_architecture is None
        ):
            raise HTTPException(
                status_code=409,
                detail="архитектура уже активна для этого датасета"
            )

        has_labels = dataset_has_labels(dataset.path)
        if not has_labels:
            dataset.current_model_architecture = body.architecture
            dataset.pending_model_architecture = None
            session.commit()
            invalidate_annotator(dataset_id)
            return {"status": "ok", "architecture": body.architecture}

        dataset.pending_model_architecture = body.architecture

        job = TrainingJob(
            dataset_id=dataset.id,
            status="queued",
            job_type="change_model_retrain"
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
            detail="не удалось поставить retrain смены архитектуры в очередь"
        )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "target_architecture": body.architecture
        }
    )




@router.get("/api/datasets/{dataset_id}/hyperparams")
async def get_hyperparams(dataset_id: int):
    '''гиперпараметры для текущей модели датасета'''
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        config = session.query(TrainingConfig).filter(TrainingConfig.dataset_id == dataset_id).first()
        if not config:
            config = TrainingConfig(dataset_id=dataset_id)
            session.add(config)
            session.commit()
            session.refresh(config)
        return {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "imgsz": config.imgsz,
            "optimizer": config.optimizer
        }
        

@router.put("/api/datasets/{dataset_id}/hyperparams")
async def update_hyperparams(dataset_id: int, body: TrainingConfigRequest):
    '''обновить гиперпараметры для текущей модели датасета'''
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        config = session.query(TrainingConfig).filter(TrainingConfig.dataset_id == dataset_id).first()
        if not config:
            config = TrainingConfig(dataset_id=dataset_id)
            session.add(config)
        
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

        session.commit()
        return {"status": "ok"}
    

@router.get('/api/datasets/{dataset_id}/metrics')
async def get_metrics(dataset_id: int):
    '''Эндпоинт для получения метрик всех версий модели для указанного датасета (precision, recall, f1, mAP и т.д.).'''
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        
        # загружаем все версии модели для датасета, отсортированные по возрастанию версии
        model_versions = session.execute(
            select(ModelVersion)
            .where(ModelVersion.dataset_id == dataset_id)
            .order_by(ModelVersion.version.asc())
        ).scalars().all()

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
            "confusion_matrix": json.loads(v.confusion_matrix_json) if v.confusion_matrix_json else None,
            "mlflow_run_id": v.mlflow_run_id,
        }
        for v in model_versions
    ]
    
@router.get('/api/datasets/{dataset_id}/metrics/latest')
async def get_curr_metrics(dataset_id: int):
    '''Эндпоинт для получения метрик текущей активной версии модели для указанного датасета.'''
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
            raise HTTPException(status_code=404, detail='Нет Активной модели')
        
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
            "confusion_matrix": json.loads(v.confusion_matrix_json) if v.confusion_matrix_json else None,
            "mlflow_run_id": v.mlflow_run_id,
}
    


@router.get("/api/datasets/{dataset_id}/augmentation")
async def get_augmentation(dataset_id: int):
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        config = session.query(TrainingConfig).filter(TrainingConfig.dataset_id == dataset_id).first()
        if not config:
            config = TrainingConfig(dataset_id=dataset_id)
            session.add(config)
            session.commit()
            session.refresh(config)
        return {
            "augmentation_enabled": config.augmentation_enabled,
            "augmentation_threshold": config.augmentation_threshold
        }
    

@router.put("/api/datasets/{dataset_id}/augmentation")
async def update_augmentation(dataset_id: int, body: AugmentationConfigRequest):
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        config = session.query(TrainingConfig).filter(TrainingConfig.dataset_id == dataset_id).first()
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