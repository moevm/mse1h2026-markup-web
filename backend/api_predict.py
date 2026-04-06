from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from requests import get
from activate import Session
import annotator
from helper import get_annotator
from typing import List
import json
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


@router.post("/api/train/{dataset_name}")
async def train(dataset_name: str):
    """Эндпоинт для запуска обучения модели на указанном датасете. Возвращает версию обученной модели."""
    dataset = get_dataset_by_name(dataset_name)
    images_dir = dataset.path

    if not os.path.exists(images_dir):
        raise HTTPException(
            status_code=404, detail=f"изображения датасета {dataset_name} не найдены"
        )

    # загружаем аннотатор (кешированный или новый) для данного датасета
    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    # запускаем обучение с гиперпараметрами из БД, сохраняем веса и метрики
    _, version = _train_and_save(dataset, annotator)
    return {"status": "ok", "dataset": dataset_name, "model_version": version}


@router.post("/api/correct/{dataset_name}")
async def correct(dataset_name: str, labeled_images: List[LabeledImage]):
    """Эндпоинт для сохранения исправленных пользователем аннотаций и дообучения модели на скорректированных данных."""
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

    # дообучаем модель на обновлённых метках и сохраняем новую версию весов
    _, version = _train_and_save(dataset, annotator)
    return {"status": "ok", "dataset": dataset_name, "model_version": version}


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
