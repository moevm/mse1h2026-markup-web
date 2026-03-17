from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from activate import Session
from db import Base, Dataset, DatasetStatus, TrainingConfig
from model_registry import get_all_models, get_model_by_id
from helper import invalidate_annotator
import os
from typing import Optional

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
                "current_model_architecture": ds.current_model_architecture
            })
        return result


@router.post("/api/addDataset")
async def add_dataset(body: AddDatasetRequest):
    '''добавить новый датасет по пути на диске'''

    # проверяем что путь существует
    if not os.path.exists(body.path):
        raise HTTPException(status_code=400, detail="указанный путь не существует")

    # считаем количество изображений
    total = 0
    for root, dirs, files in os.walk(body.path):
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
            path=body.path,
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
            "current_model_architecture": dataset.current_model_architecture
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


@router.post("/api/datasets/{dataset_id}/model")
async def change_model(dataset_id: int, body: ChangeModelRequest):
    '''сменить архитектуру модели для датасета'''

    # проверяем что такая модель существует в реестре
    model_info = get_model_by_id(body.architecture)
    if not model_info:
        raise HTTPException(status_code=400, detail=f"неизвестная архитектура: {body.architecture}")

    # обновляем в бд
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет не найден")
        dataset.current_model_architecture = body.architecture
        session.commit()

    # сбрасываем кеш чтобы следующий predict загрузил новую модель
    invalidate_annotator(dataset_id)

    return {"status": "ok", "architecture": body.architecture}



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