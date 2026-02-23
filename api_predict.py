from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from annotator import AutoAnnotator
from typing import List
from PIL import Image, UnidentifiedImageError
import io
import os

app = FastAPI()
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


@app.post("/upload/{dataset_name}")
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


@app.post("/predict/{dataset_name}")
async def predict(dataset_name: str, filenames: List[str]):
    '''путь к файлам'''
    images_dir = get_images_dir(dataset_name)
    response = []
    for filename in filenames:
        image_path = os.path.join(images_dir, filename)
        '''проверка на существование'''
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"{filename} -  файл не найден в {dataset_name}")
        image = Image.open(image_path)
        '''предикты'''
        annotations = annotator.predict(image)
        response.append({"filename": filename, "annotations": annotations})
    return response


@app.post("/train/{dataset_name}")
async def train(dataset_name: str, labeled_images: List[LabeledImage]):
    '''сохранение меток и запуск обучения'''

    # проверка что датасет существует
    images_dir = os.path.join("datasets", dataset_name, "images", "train")
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail=f"датасет: {dataset_name} не найден")

    # сохранение txt меток для каждого файла
    for item in labeled_images:
        image_path = os.path.join(images_dir, item.filename)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"{item.filename} не найден в {dataset_name}")
        annotator.save_labels(dataset_name, item.filename, [ann.dict() for ann in item.annotations])

    # запуск обучения
    annotator.train(dataset_name)
    return {"status": "ok", "dataset": dataset_name}
