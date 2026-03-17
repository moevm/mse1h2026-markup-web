from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from typing import Optional
from datetime import datetime, timezone


class Base(DeclarativeBase):
    pass


class DatasetStatus(Base):
    __tablename__ = "dataset_status"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))


class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(primary_key = True)
    name: Mapped[str] = mapped_column(String(255))
    status_id: Mapped[int] = mapped_column(ForeignKey("dataset_status.id"))
    total_size: Mapped[int] = mapped_column()
    inwork_size: Mapped[int] = mapped_column()
    path: Mapped[str] = mapped_column(String(255))
    average_percent_success: Mapped[Optional[float]] = mapped_column(nullable=True)
    current_model_architecture: Mapped[str] = mapped_column(String(100), default="yolo11n") # удет хранить какую архитектуру сейчас использует данный датасет


class ModelVersion(Base):
    __tablename__ = "model_version"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    version: Mapped[int] = mapped_column()
    path: Mapped[str] = mapped_column(String(255))
    epochs: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    is_active: Mapped[bool] = mapped_column(default=True)
    architecture: Mapped[str] = mapped_column(String(100)) # тобы каждая сохранённая версия знала от какой архитектуры она произошла




