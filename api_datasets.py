from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from activate import Session
from db import Dataset, DatasetStatus
import os

router = APIRouter()


class AddDatasetRequest(BaseModel):
    dataset_name: str
    path: str


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
                "average_percent_success": ds.average_percent_success
            })
        return result


@router.post("/api/addDataset")
async def add_dataset(body: AddDatasetRequest):
    '''добавить новый датасет по пути на диске'''

    # проверяем что путь существует
    if not os.path.exists(body.path):
        raise HTTPException(status_code=400, detail="указанный путь не существует")

    # считаем количество изображений
    total = len([
        f for f in os.listdir(body.path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

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
            "average_percent_success": None
        }
