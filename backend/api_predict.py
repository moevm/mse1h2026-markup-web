from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from activate import Session
from helper import get_annotator, invalidate_annotator
from typing import List
import json
from PIL import Image, UnidentifiedImageError
from training import _train_and_save
from db import Dataset
import io
import os


router = APIRouter()


# модели запроса
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
    '''создание директорий под датасеты'''
    path = os.path.join("datasets", dataset_name, "images", "train")
    os.makedirs(path, exist_ok=True)
    return path


def get_dataset_by_name(dataset_name: str) -> Dataset:
    '''находим датасет в бд по имени, кидаем 404 если не найден'''
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.name == dataset_name).first()
        if not dataset:
            raise HTTPException(status_code=404, detail=f"датасет: {dataset_name} не найден")
        # чтобы объект жил за пределами сессии запоминаем нужные поля
        session.expunge(dataset)
        return dataset

@router.post("/upload/{dataset_name}")
async def upload(dataset_name: str, files: List[UploadFile] = File(...)):
    '''загрузка конкретного набора файлов'''
    get_dataset_by_name(dataset_name)
    images_dir = get_images_dir(dataset_name)
    saved = []

    for file in files:
        contents = await file.read()
        '''проверка на корректность данных'''
        try:
            Image.open(io.BytesIO(contents)).verify()
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"{file.filename} - не явл. картинкой")
        save_path = os.path.join(images_dir, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        saved.append({"filename": file.filename, "path": save_path})

    return {"dataset": dataset_name, "uploaded": saved}


@router.post("/train/{dataset_name}")
async def train(dataset_name: str):
    '''дообучение на уже сохранённых метках'''
    dataset = get_dataset_by_name(dataset_name)

    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"изображения датасета {dataset_name} не найдены")

    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    _, version = _train_and_save(dataset, dataset_name, annotator)
    return {"status": "ok", "dataset": dataset_name, "model_version": version}


@router.post("/correct/{dataset_name}")
async def correct(dataset_name: str, labeled_images: List[LabeledImage]):
    '''сохранение исправленных аннотаций + дообучение'''
    dataset = get_dataset_by_name(dataset_name)

    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"изображения датасета {dataset_name} не найдены")

    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    for item in labeled_images:
        image_path = os.path.join(images_dir, item.filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"{item.filename} не найден в {dataset_name}")
        annotator.save_labels(dataset_name, item.filename, [ann.model_dump() for ann in item.annotations])

    _, version = _train_and_save(dataset, dataset_name, annotator)
    return {"status": "ok", "dataset": dataset_name, "model_version": version}


@router.get("/datasets/{dataset_name}/images")
async def list_images(dataset_name: str):
    '''список изображений датасета'''
    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="датасет не найден")
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return {"dataset": dataset_name, "images": files}


@router.get("/datasets/{dataset_name}/images/{filename}")
async def get_image(dataset_name: str, filename: str):
    '''отдать конкретное изображение'''
    image_path = os.path.join("datasets", dataset_name, "images", "train", filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="файл не найден")
    return FileResponse(image_path)


