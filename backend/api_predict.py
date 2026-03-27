from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from requests import get
from activate import Session
import annotator
from helper import get_annotator, invalidate_annotator
from typing import List
import json
from PIL import Image, UnidentifiedImageError
from training import _train_and_save
from db import Dataset
import io
import os

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


def get_images_dir(dataset_name: str) -> str:
    path = os.path.join("datasets", dataset_name, "images", "train")
    os.makedirs(path, exist_ok=True)
    return path


def get_dataset_by_name(dataset_name: str) -> Dataset:
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.name == dataset_name).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"датасет: {dataset_name} не найден")
        session.expunge(dataset)
        return dataset


@router.post("/api/upload/{dataset_name}")
async def upload(dataset_name: str, files: List[UploadFile] = File(...)):
    get_dataset_by_name(dataset_name)
    images_dir = get_images_dir(dataset_name)
    saved = []

    for file in files:
        contents = await file.read()
        try:
            Image.open(io.BytesIO(contents)).verify()
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"{file.filename} - не явл. картинкой")
        
        # Безопасное имя файла
        safe_filename = os.path.basename(file.filename)
        save_path = os.path.join(images_dir, safe_filename)
        
        with open(save_path, "wb") as f:
            f.write(contents)
        saved.append({"filename": safe_filename, "path": save_path})

    return {"dataset": dataset_name, "uploaded": saved}


@router.post("/api/train/{dataset_name}")
async def train(dataset_name: str):
    dataset = get_dataset_by_name(dataset_name)
    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"изображения датасета {dataset_name} не найдены")

    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    _, version = _train_and_save(dataset, dataset_name, annotator)
    return {"status": "ok", "dataset": dataset_name, "model_version": version}


@router.post("/api/correct/{dataset_name}")
async def correct(dataset_name: str, labeled_images: List[LabeledImage]):
    dataset = get_dataset_by_name(dataset_name)
    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"изображения датасета {dataset_name} не найдены")

    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    for item in labeled_images:
        safe_filename = os.path.basename(item.filename)
        image_path = os.path.join(images_dir, safe_filename)
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"{safe_filename} не найден в {dataset_name}")
        
        annotator.save_labels(dataset_name, safe_filename, [ann.model_dump() for ann in item.annotations])

    _, version = _train_and_save(dataset, dataset_name, annotator)
    return {"status": "ok", "dataset": dataset_name, "model_version": version}


@router.get("/api/datasets/{dataset_name}/images")
async def list_images(dataset_name: str):
    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="датасет не найден")
    
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return {"dataset": dataset_name, "images": files}


@router.get("/api/datasets/{dataset_name}/images/{filename}")
async def get_image(dataset_name: str, filename: str):
    safe_filename = os.path.basename(filename)
    image_path = os.path.join("datasets", dataset_name, "images", "train", safe_filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="файл не найден")
    return FileResponse(image_path)


@router.post('/api/predict/{dataset_name}/{filename}')
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
    image_path = os.path.join("datasets", dataset_name, "images", "train", safe_filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="файл не найден")
    
    boxes = annotator.predict(image_path)
    img_w, img_h = Image.open(image_path).size

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