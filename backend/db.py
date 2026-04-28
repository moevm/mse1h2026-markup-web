from sqlalchemy import ForeignKey
from sqlalchemy import String, Text
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
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    status_id: Mapped[int] = mapped_column(ForeignKey("dataset_status.id"))
    total_size: Mapped[int] = mapped_column()
    inwork_size: Mapped[int] = mapped_column()
    path: Mapped[str] = mapped_column(String(255))
    average_percent_success: Mapped[Optional[float]] = mapped_column(nullable=True)
    current_model_architecture: Mapped[str] = mapped_column(String(100), default="yolo11n")
    pending_model_architecture: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    classes_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metric_precision: Mapped[Optional[float]] = mapped_column(nullable=True)
    metric_recall: Mapped[Optional[float]] = mapped_column(nullable=True)
    metric_f1: Mapped[Optional[float]] = mapped_column(nullable=True)
    metric_mean_iou: Mapped[Optional[float]] = mapped_column(nullable=True)
    metrics_boxes_total: Mapped[int] = mapped_column(default=0)
    metrics_images_total: Mapped[int] = mapped_column(default=0)


class ModelVersion(Base):
    __tablename__ = "model_version"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    version: Mapped[int] = mapped_column()
    path: Mapped[str] = mapped_column(String(255))
    epochs: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    is_active: Mapped[bool] = mapped_column(default=True)
    architecture: Mapped[str] = mapped_column(String(100))
    precision: Mapped[float] = mapped_column(nullable=True)
    recall: Mapped[float] = mapped_column(nullable=True)
    f1: Mapped[Optional[float]] = mapped_column(nullable=True)
    map50: Mapped[float] = mapped_column(nullable=True)
    map50_95: Mapped[float] = mapped_column(nullable=True)
    mean_iou: Mapped[float] = mapped_column(nullable=True)
    confusion_matrix_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


class TrainingConfig(Base):
    __tablename__ = "training_config"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    epochs: Mapped[int] = mapped_column(default=10)
    batch_size: Mapped[int] = mapped_column(default=16)
    learning_rate: Mapped[float] = mapped_column(default=0.001)
    imgsz: Mapped[int] = mapped_column(default=640)
    optimizer: Mapped[str] = mapped_column(String(50), default="AdamW")
    augmentation_enabled: Mapped[bool] = mapped_column(default=True)
    augmentation_threshold: Mapped[float] = mapped_column(default=0.85)


class TrainingJob(Base):
    __tablename__ = "training_job"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    status: Mapped[str] = mapped_column(String(32))
    job_type: Mapped[str] = mapped_column(String(64), default="train")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    model_version_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model_version.id"), nullable=True)


class ImageStatus(Base):
    __tablename__ = "image_status"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    code: Mapped[str] = mapped_column(String(50), unique=True)


class DatasetImage(Base):
    __tablename__ = "dataset_image"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    filename: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    label_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status_id: Mapped[int] = mapped_column(ForeignKey("image_status.id"), default=1)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))


class PredictionBatch(Base):
    __tablename__ = "prediction_batch"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    model_version_id: Mapped[Optional[int]] = mapped_column(ForeignKey("model_version.id"), nullable=True)
    architecture: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(32), default="active")
    image_filenames_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


class PredictionBox(Base):
    __tablename__ = "prediction_box"
    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("prediction_batch.id"))
    image_filename: Mapped[str] = mapped_column(String(255))
    class_id: Mapped[int] = mapped_column()
    confidence: Mapped[float] = mapped_column()
    x1: Mapped[int] = mapped_column()
    y1: Mapped[int] = mapped_column()
    x2: Mapped[int] = mapped_column()
    y2: Mapped[int] = mapped_column()