from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from activate import Session
from helper import get_annotator
from typing import List
from db import Dataset, TrainingJob
from job_runner import submit_training_job
from datetime import datetime, timezone
import os
from PIL import Image

router = APIRouter()


# Модели запроса
class AnnotationItem(BaseModel):
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


class LabeledImage(BaseModel):
    filename: str
    annotations: List[AnnotationItem]


def get_dataset_by_name(dataset_name: str) -> Dataset:
    """Получение объекта датасета из БД по имени. Выбрасывает 404, если датасет не найден."""
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.name == dataset_name).first()
        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"датасет: {dataset_name} не найден"
            )
        session.expunge(dataset)
        return dataset


@router.post("/api/train/{dataset_name}", status_code=status.HTTP_202_ACCEPTED)
async def train(dataset_name: str):
    """Эндпоинт для постановки обучения модели в очередь для указанного датасета."""
    dataset = get_dataset_by_name(dataset_name)
    images_dir = dataset.path

    if not os.path.exists(images_dir):
        raise HTTPException(
            status_code=404, detail=f"изображения датасета {dataset_name} не найдены"
        )

    with Session() as session:
        session.query(Dataset).filter(Dataset.id == dataset.id).with_for_update().first()

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
                detail=f"обучение для датасета {dataset_name} уже запущено"
            )

        job = TrainingJob(dataset_id=dataset.id, status="queued")
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    try:
        submit_training_job(job_id, dataset.id)
    except Exception as exc:
        with Session() as session:
            failed_job = session.get(TrainingJob, job_id)
            if failed_job:
                failed_job.status = "failed"
                failed_job.error = f"failed to submit job: {exc}"
                failed_job.finished_at = datetime.now(timezone.utc)
                session.commit()
        raise HTTPException(
            status_code=500,
            detail="не удалось поставить обучение в очередь"
        )

    return {"job_id": job_id, "status": "queued"}


@router.get("/api/train/jobs/{job_id}")
async def get_training_job(job_id: int):
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"job id={job_id} не найден")

        return {
            "id": job.id,
            "dataset_id": job.dataset_id,
            "status": job.status,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }


@router.post("/api/correct/{dataset_name}")
async def correct(dataset_name: str, labeled_images: List[LabeledImage]):
    """Эндпоинт для сохранения исправленных пользователем аннотаций."""
    dataset = get_dataset_by_name(dataset_name)
    images_dir = dataset.path

    if not os.path.exists(images_dir):
        raise HTTPException(
            status_code=404, detail=f"изображения датасета {dataset_name} не найдены"
        )

    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    # перебираем присланные изображения, сохраняем исправленные метки в YOLO-формате
    for item in labeled_images:
        # защита от path traversal — берём только имя файла
        safe_filename = os.path.basename(item.filename)
        image_path = os.path.join(images_dir, safe_filename)

        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404, detail=f"{safe_filename} не найден в {dataset_name}"
            )

        # конвертируем аннотации в dict и записываем в labels/train/<имя>.txt
        annotator.save_labels(
            dataset.path, safe_filename, [ann.model_dump() for ann in item.annotations]
        )

    return {"status": "ok", "dataset": dataset_name}


@router.get("/api/datasets/{dataset_name}/images")
async def list_images(dataset_name: str):
    """Эндпоинт для получения списка всех изображений в указанном датасете."""
    dataset = get_dataset_by_name(dataset_name)
    images_dir = dataset.path
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="датасет не найден")

    # фильтруем только файлы изображений по расширению
    files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return {"dataset": dataset_name, "images": files}


@router.get("/api/datasets/{dataset_name}/images/{filename}")
async def get_image(dataset_name: str, filename: str):
    """Эндпоинт для получения конкретного изображения из датасета по имени файла."""
    dataset = get_dataset_by_name(dataset_name)
    # защита от path traversal — берём только имя файла без директорий
    safe_filename = os.path.basename(filename)
    image_path = os.path.join(dataset.path, safe_filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="файл не найден")
    # отдаём файл как бинарный ответ с автоопределением content-type
    return FileResponse(image_path)


@router.post("/api/predict/{dataset_name}/{filename}")
async def predict(dataset_name: str, filename: str):
    """
    Предсказание объектов на изображении.
    Возвращает список bounding boxes с метками и координатами.
    """
    dataset = get_dataset_by_name(dataset_name)
    annotator = get_annotator(dataset.id)

    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    safe_filename = os.path.basename(filename)
    image_path = os.path.join(dataset.path, safe_filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="файл не найден")

    boxes = annotator.predict(image_path)
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    return [
        {
            "id": i,
            "label": b["class_name"],
            "conf": b["confidence"],
            "x": b["x1"] / img_w * 100,
            "y": b["y1"] / img_h * 100,
            "w": (b["x2"] - b["x1"]) / img_w * 100,
            "h": (b["y2"] - b["y1"]) / img_h * 100,
        }
        for i, b in enumerate(boxes)
    ]
