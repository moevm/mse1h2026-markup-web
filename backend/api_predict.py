from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from annotator import AutoAnnotator
from activate import Session
from db import ModelVersion
from typing import List
from PIL import Image, UnidentifiedImageError
import io
import os

router = APIRouter()
annotator = AutoAnnotator()


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
    '''функция - создание директорий под датасеты'''
    path = os.path.join("datasets", dataset_name, "images", "train")
    os.makedirs(path, exist_ok=True)
    return path


@router.post("/upload/{dataset_name}")
async def upload(dataset_name: str, files: List[UploadFile] = File(...)):
    '''загрузка конкретного набора файлов'''
    images_dir = get_images_dir(dataset_name)
    saved = []
    '''итерация по файлам'''
    for file in files:
        '''считывание'''
        contents = await file.read()
        '''проверка на корректность данных'''
        try:
            Image.open(io.BytesIO(contents)).verify()
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"{file.filename} - не явл. картинкой")
        save_path = os.path.join(images_dir, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        '''сохранение'''
        saved.append({"filename": file.filename, "path": save_path})
    return {"dataset": dataset_name, "uploaded": saved}


@router.post("/train/{dataset_name}")
async def train(dataset_name: str, filenames: List[str]):
    '''дообучение на уже сохранённых метках'''
    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"датасет: {dataset_name} не найден")

    model_path, version = annotator.train(dataset_name)

    with Session() as session:
        model_record = ModelVersion(
            dataset_id=1,  # TODO: заменить на реальный id из бд = заглушка
            version=version,
            path=model_path,
            epochs=10,
            is_active=True
        )
        session.add(model_record)
        session.commit()

    return {"status": "ok", "dataset": dataset_name}


@router.post("/correct/{dataset_name}")
async def correct(dataset_name: str, labeled_images: List[LabeledImage]):
    '''сохранение исправленных аннотаций + дообучение'''
    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"датасет: {dataset_name} не найден")

    for item in labeled_images:
        image_path = os.path.join(images_dir, item.filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"{item.filename} не найден в {dataset_name}")
        annotator.save_labels(dataset_name, item.filename, [ann.model_dump() for ann in item.annotations])

    model_path, version = annotator.train(dataset_name)

    with Session() as session:
        model_record = ModelVersion(
            dataset_id=1,  # TODO: заменить на реальный id из бд
            version=version,
            path=model_path,
            epochs=10,
            is_active=True
        )
        session.add(model_record)
        session.commit()

    return {"status": "ok", "dataset": dataset_name, "model_version": version}





@router.get("/datasets/{dataset_name}/images")
async def list_images(dataset_name: str):
    images_dir = os.path.join("datasets", dataset_name, "images", "train") 
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="датасет не найден")
    files = [f for f in os.listdir(images_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return {"dataset": dataset_name, "images": files}



@router.get("/datasets/{dataset_name}/images/{filename}")
async def get_image(dataset_name: str, filename: str):
    '''отдать конкретное изображение'''
    image_path = os.path.join(get_images_dir(dataset_name), filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="файл не найден")
    return FileResponse(image_path)

