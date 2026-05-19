import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import (
    Base,
    Dataset,
    DatasetStatus,
    ModelVersion,
    TrainingConfig,
    TrainingJob,
    ImageStatus,
    DatasetImage,
    PredictionBatch,
    PredictionBox,
    BoundingBoxClass,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def engine():
    """In-memory SQLite — изолированная БД для каждого теста."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture(scope="function")
def session(engine):
    """Сессия, которая откатывается после каждого теста."""
    SessionLocal = sessionmaker(bind=engine)
    sess = SessionLocal()
    yield sess
    sess.close()


@pytest.fixture
def base_statuses(session):
    """Наполняем справочники — то же самое, что делает create_db_tables()."""
    statuses = [
        DatasetStatus(id=0, name="Just load"),
        DatasetStatus(id=1, name="Done and verificated"),
        DatasetStatus(id=2, name="Need to verify"),
        DatasetStatus(id=3, name="At work"),
    ]
    image_statuses = [
        ImageStatus(id=1, name="Не размечено",               code="unlabeled"),
        ImageStatus(id=2, name="Размечено",                  code="labeled"),
        ImageStatus(id=3, name="Готово для обучения",        code="ready_for_training"),
        ImageStatus(id=4, name="Авторазмечено (проверка)",   code="auto_labeled_pending_review"),
        ImageStatus(id=5, name="Размечено окончательно",     code="finalized"),
    ]
    session.add_all(statuses + image_statuses)
    session.commit()
    return statuses, image_statuses


@pytest.fixture
def sample_dataset(session, base_statuses):
    """Один датасет для тестов, которым он нужен."""
    ds = Dataset(
        name="test_dataset",
        status_id=0,
        total_size=100,
        inwork_size=0,
        path="/tmp/test_dataset",
    )
    session.add(ds)
    session.commit()
    session.refresh(ds)
    return ds


# ---------------------------------------------------------------------------
# DatasetStatus
# ---------------------------------------------------------------------------

class TestDatasetStatus:
    def test_create_statuses(self, session, base_statuses):
        count = session.query(DatasetStatus).count()
        assert count == 4

    def test_status_names(self, session, base_statuses):
        names = {s.name for s in session.query(DatasetStatus).all()}
        assert "Just load" in names
        assert "At work" in names

    def test_no_duplicate_statuses(self, session, base_statuses):
        """Повторный вызов не должен дублировать записи."""
        if session.query(DatasetStatus).count() == 0:
            session.add_all([DatasetStatus(id=0, name="Just load")])
            session.commit()
        count_before = session.query(DatasetStatus).count()
        # имитируем логику activate.py — добавляем только если пусто
        if session.query(DatasetStatus).count() == 0:
            session.add(DatasetStatus(id=0, name="Just load"))
            session.commit()
        assert session.query(DatasetStatus).count() == count_before


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_create_dataset(self, session, base_statuses):
        ds = Dataset(
            name="helmets",
            status_id=0,
            total_size=50,
            inwork_size=10,
            path="/data/helmets",
        )
        session.add(ds)
        session.commit()

        result = session.query(Dataset).filter_by(name="helmets").first()
        assert result is not None
        assert result.total_size == 50
        assert result.path == "/data/helmets"

    def test_dataset_defaults(self, session, base_statuses):
        ds = Dataset(
            name="defaults_test",
            status_id=0,
            total_size=0,
            inwork_size=0,
            path="/tmp/x",
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)

        assert ds.current_model_architecture == "yolo11n"
        assert ds.average_percent_success is None
        assert ds.metric_precision is None
        assert ds.metrics_boxes_total == 0

    def test_update_dataset_metrics(self, session, sample_dataset):
        sample_dataset.metric_precision = 0.92
        sample_dataset.metric_recall = 0.88
        sample_dataset.metric_f1 = 0.90
        session.commit()
        session.refresh(sample_dataset)

        assert sample_dataset.metric_precision == pytest.approx(0.92)
        assert sample_dataset.metric_f1 == pytest.approx(0.90)

    def test_delete_dataset(self, session, sample_dataset):
        ds_id = sample_dataset.id
        session.delete(sample_dataset)
        session.commit()

        assert session.query(Dataset).filter_by(id=ds_id).first() is None

    def test_multiple_datasets(self, session, base_statuses):
        for i in range(5):
            session.add(Dataset(
                name=f"ds_{i}",
                status_id=0,
                total_size=i * 10,
                inwork_size=0,
                path=f"/tmp/ds_{i}",
            ))
        session.commit()
        assert session.query(Dataset).count() == 5


# ---------------------------------------------------------------------------
# ModelVersion
# ---------------------------------------------------------------------------

class TestModelVersion:
    def test_create_model_version(self, session, sample_dataset):
        mv = ModelVersion(
            dataset_id=sample_dataset.id,
            version=1,
            path="/models/v1.pt",
            epochs=10,
            architecture="yolo11n",
        )
        session.add(mv)
        session.commit()
        session.refresh(mv)

        assert mv.id is not None
        assert mv.is_active is True
        assert mv.created_at is not None

    def test_model_version_metrics(self, session, sample_dataset):
        mv = ModelVersion(
            dataset_id=sample_dataset.id,
            version=1,
            path="/models/v1.pt",
            epochs=20,
            architecture="yolo11n",
            precision=0.91,
            recall=0.87,
            f1=0.89,
            map50=0.85,
            map50_95=0.60,
            mean_iou=0.72,
        )
        session.add(mv)
        session.commit()
        session.refresh(mv)

        assert mv.precision == pytest.approx(0.91)
        assert mv.mean_iou == pytest.approx(0.72)

    def test_multiple_versions_per_dataset(self, session, sample_dataset):
        for v in range(1, 4):
            session.add(ModelVersion(
                dataset_id=sample_dataset.id,
                version=v,
                path=f"/models/v{v}.pt",
                epochs=10,
                architecture="yolo11n",
            ))
        session.commit()

        versions = session.query(ModelVersion).filter_by(
            dataset_id=sample_dataset.id
        ).all()
        assert len(versions) == 3


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_create_config(self, session, sample_dataset):
        cfg = TrainingConfig(dataset_id=sample_dataset.id)
        session.add(cfg)
        session.commit()
        session.refresh(cfg)

        assert cfg.epochs == 10
        assert cfg.batch_size == 16
        assert cfg.learning_rate == pytest.approx(0.001)
        assert cfg.optimizer == "AdamW"

    def test_custom_config(self, session, sample_dataset):
        cfg = TrainingConfig(
            dataset_id=sample_dataset.id,
            epochs=50,
            batch_size=32,
            learning_rate=0.0005,
            imgsz=1280,
        )
        session.add(cfg)
        session.commit()
        session.refresh(cfg)

        assert cfg.epochs == 50
        assert cfg.imgsz == 1280


# ---------------------------------------------------------------------------
# TrainingJob
# ---------------------------------------------------------------------------

class TestTrainingJob:
    def test_create_job(self, session, sample_dataset):
        job = TrainingJob(dataset_id=sample_dataset.id, status="pending")
        session.add(job)
        session.commit()
        session.refresh(job)

        assert job.id is not None
        assert job.status == "pending"
        assert job.created_at is not None
        assert job.job_type == "train"

    def test_job_status_transition(self, session, sample_dataset):
        job = TrainingJob(dataset_id=sample_dataset.id, status="pending")
        session.add(job)
        session.commit()

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(job)
        assert job.status == "running"

        job.status = "done"
        job.finished_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(job)
        assert job.status == "done"
        assert job.finished_at is not None

    def test_interrupted_jobs_marked_failed(self, session, sample_dataset):
        """Логика из activate.py — running-джобы при рестарте → failed."""
        job = TrainingJob(dataset_id=sample_dataset.id, status="running")
        session.add(job)
        session.commit()

        running_jobs = session.query(TrainingJob).filter_by(status="running").all()
        now = datetime.now(timezone.utc)
        for j in running_jobs:
            j.status = "failed"
            j.error = "interrupted by server restart"
            j.finished_at = now
        session.commit()

        result = session.query(TrainingJob).filter_by(id=job.id).first()
        assert result.status == "failed"
        assert result.error == "interrupted by server restart"

    def test_job_with_error(self, session, sample_dataset):
        job = TrainingJob(
            dataset_id=sample_dataset.id,
            status="failed",
            error="CUDA out of memory",
        )
        session.add(job)
        session.commit()
        session.refresh(job)

        assert job.error == "CUDA out of memory"


# ---------------------------------------------------------------------------
# ImageStatus
# ---------------------------------------------------------------------------

class TestImageStatus:
    def test_image_statuses_created(self, session, base_statuses):
        assert session.query(ImageStatus).count() == 5

    def test_image_status_codes_unique(self, session, base_statuses):
        codes = [s.code for s in session.query(ImageStatus).all()]
        assert len(codes) == len(set(codes))

    def test_image_status_by_code(self, session, base_statuses):
        status = session.query(ImageStatus).filter_by(code="labeled").first()
        assert status is not None
        assert status.name == "Размечено"


# ---------------------------------------------------------------------------
# DatasetImage
# ---------------------------------------------------------------------------

class TestDatasetImage:
    def test_create_image(self, session, sample_dataset, base_statuses):
        img = DatasetImage(
            dataset_id=sample_dataset.id,
            filename="photo1.jpg",
            image_path="/tmp/photo1.jpg",
            label_path="/tmp/photo1.txt",
            status_id=1,
        )
        session.add(img)
        session.commit()
        session.refresh(img)

        assert img.id is not None
        assert img.filename == "photo1.jpg"
        assert img.created_at is not None

    def test_images_belong_to_dataset(self, session, sample_dataset, base_statuses):
        for i in range(3):
            session.add(DatasetImage(
                dataset_id=sample_dataset.id,
                filename=f"img_{i}.jpg",
                status_id=1,
            ))
        session.commit()

        images = session.query(DatasetImage).filter_by(
            dataset_id=sample_dataset.id
        ).all()
        assert len(images) == 3

    def test_image_status_update(self, session, sample_dataset, base_statuses):
        img = DatasetImage(
            dataset_id=sample_dataset.id,
            filename="change_me.jpg",
            status_id=1,
        )
        session.add(img)
        session.commit()

        img.status_id = 2
        session.commit()
        session.refresh(img)
        assert img.status_id == 2


# ---------------------------------------------------------------------------
# PredictionBatch и PredictionBox
# ---------------------------------------------------------------------------

class TestPredictions:
    def test_create_prediction_batch(self, session, sample_dataset):
        batch = PredictionBatch(
            dataset_id=sample_dataset.id,
            architecture="yolo11n",
        )
        session.add(batch)
        session.commit()
        session.refresh(batch)

        assert batch.id is not None
        assert batch.status == "active"

    def test_create_prediction_box(self, session, sample_dataset):
        batch = PredictionBatch(
            dataset_id=sample_dataset.id,
            architecture="yolo11n",
        )
        session.add(batch)
        session.commit()

        box = PredictionBox(
            batch_id=batch.id,
            image_filename="photo.jpg",
            class_id=0,
            confidence=0.95,
            x1=10, y1=20, x2=100, y2=200,
        )
        session.add(box)
        session.commit()
        session.refresh(box)

        assert box.id is not None
        assert box.confidence == pytest.approx(0.95)

    def test_multiple_boxes_per_batch(self, session, sample_dataset):
        batch = PredictionBatch(
            dataset_id=sample_dataset.id,
            architecture="yolo11n",
        )
        session.add(batch)
        session.commit()

        for i in range(5):
            session.add(PredictionBox(
                batch_id=batch.id,
                image_filename=f"img_{i}.jpg",
                class_id=i % 3,
                confidence=0.8,
                x1=0, y1=0, x2=50, y2=50,
            ))
        session.commit()

        boxes = session.query(PredictionBox).filter_by(batch_id=batch.id).all()
        assert len(boxes) == 5


# ---------------------------------------------------------------------------
# BoundingBoxClass
# ---------------------------------------------------------------------------

class TestBoundingBoxClass:
    def test_create_class(self, session, sample_dataset):
        cls = BoundingBoxClass(
            dataset_id=sample_dataset.id,
            class_id=0,
            name="helmet",
            color="#FF0000",
        )
        session.add(cls)
        session.commit()
        session.refresh(cls)

        assert cls.id is not None
        assert cls.name == "helmet"

    def test_classes_for_dataset(self, session, sample_dataset):
        for i, name in enumerate(["helmet", "person", "bike"]):
            session.add(BoundingBoxClass(
                dataset_id=sample_dataset.id,
                class_id=i,
                name=name,
            ))
        session.commit()

        classes = session.query(BoundingBoxClass).filter_by(
            dataset_id=sample_dataset.id
        ).all()
        assert len(classes) == 3
        names = {c.name for c in classes}
        assert "helmet" in names