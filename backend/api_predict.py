from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from activate import Session
from helper import get_annotator
from db import Dataset, DatasetImage, PredictionBox, TrainingJob
from job_runner import submit_training_job
from typing import List
import json
import os

router = APIRouter()


def get_dataset_by_name(dataset_name: str) -> Dataset:
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.name == dataset_name).first()
        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"Датасет: {dataset_name} не найден"
            )
        session.expunge(dataset)
        return dataset


@router.post("/api/train/{dataset_name}", status_code=status.HTTP_202_ACCEPTED)
def train(dataset_name: str):
    dataset = get_dataset_by_name(dataset_name)
    images_dir = dataset.path
    if not os.path.exists(images_dir):
        raise HTTPException(
            status_code=404, detail=f"Изображения датасета {dataset_name} не найдены"
        )

    with Session() as session:
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
                detail=f"Обучение для датасета {dataset_name} уже запущено"
            )

        job = TrainingJob(dataset_id=dataset.id, status="queued", job_type="train")
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    try:
        submit_training_job(job_id)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Не удалось поставить обучение в очередь"
        )

    return {"job_id": job_id, "status": "queued"}


@router.get("/api/train/jobs/{job_id}")
def get_training_job(job_id: int):
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job id={job_id} не найден")

        return {
            "id": job.id,
            "dataset_id": job.dataset_id,
            "status": job.status,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }


@router.post("/api/predict/{dataset_name}/{filename}")
async def predict(dataset_name: str, filename: str):
    dataset = get_dataset_by_name(dataset_name)
    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="Не удалось загрузить модель")

    safe_filename = os.path.basename(filename)
    image_path = os.path.join(dataset.path, "images", safe_filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Файл не найден")

    boxes = annotator.predict(image_path)

    return [
        {
            "id": i,
            "class_id": b["class_id"],
            "label": b["class_name"],
            "conf": b["confidence"],
            "x1": int(b["x1"]),
            "y1": int(b["y1"]),
            "x2": int(b["x2"]),
            "y2": int(b["y2"]),
        }
        for i, b in enumerate(boxes)
    ]
