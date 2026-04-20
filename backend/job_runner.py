from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing

from activate import Session
from annotator import AutoAnnotator
from db import Dataset, ModelVersion, TrainingJob
from helper import get_annotator, invalidate_annotator
from model_registry import get_model_by_id
from training import _train_and_save


_executor: ProcessPoolExecutor | None = None
if multiprocessing.current_process().name == "MainProcess":
    _executor = ProcessPoolExecutor(max_workers=1)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _mark_job_failed(job_id: int, error_message: str):
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return

        job.status = "failed"
        job.error = error_message
        job.finished_at = _now_utc()
        session.commit()


def _mark_job_done(job_id: int, model_version_id: int | None = None):
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return

        job.status = "done"
        job.error = None
        job.finished_at = _now_utc()
        if model_version_id is not None:
            job.model_version_id = model_version_id
        session.commit()


def _start_job(job_id: int) -> tuple[int, str]:
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise RuntimeError(f"training job id={job_id} not found")
        if job.status != "queued":
            raise RuntimeError(f"training job id={job_id} is not queued")

        job.status = "running"
        job.started_at = _now_utc()
        job.finished_at = None
        job.error = None
        session.commit()
        return job.dataset_id, job.job_type


def _get_dataset_detached(dataset_id: int) -> Dataset:
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise RuntimeError(f"dataset with id={dataset_id} not found")
        session.expunge(dataset)
        return dataset


def _run_train_job(job_id: int, dataset_id: int):
    dataset = _get_dataset_detached(dataset_id)

    annotator = get_annotator(dataset_id)
    if not annotator:
        raise RuntimeError("не удалось загрузить модель")

    _, _, model_version_id = _train_and_save(
        dataset,
        annotator,
        activate_new_version=True
    )
    _mark_job_done(job_id, model_version_id=model_version_id)


def _run_change_model_retrain_job(job_id: int, dataset_id: int):
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise RuntimeError(f"dataset with id={dataset_id} not found")
        if not dataset.pending_model_architecture:
            raise RuntimeError("pending_model_architecture не задана")
        target_architecture = dataset.pending_model_architecture
        session.expunge(dataset)

    model_info = get_model_by_id(target_architecture)
    if not model_info:
        raise RuntimeError(f"неизвестная архитектура: {target_architecture}")

    annotator = AutoAnnotator(
        model_path=model_info["weights"],
        model_type=model_info["type"]
    )
    _, _, new_model_version_id = _train_and_save(
        dataset,
        annotator,
        activate_new_version=False,
        architecture_override=target_architecture
    )

    with Session() as session:
        dataset_row = session.get(Dataset, dataset_id)
        if not dataset_row:
            raise RuntimeError(f"dataset with id={dataset_id} not found after retrain")

        new_model_version = session.get(ModelVersion, new_model_version_id)
        if not new_model_version or new_model_version.dataset_id != dataset_id:
            raise RuntimeError("новая версия модели не найдена после retrain")

        session.query(ModelVersion).filter(
            ModelVersion.dataset_id == dataset_id,
            ModelVersion.is_active == True
        ).update({"is_active": False})

        new_model_version.is_active = True
        dataset_row.current_model_architecture = target_architecture
        dataset_row.pending_model_architecture = None
        session.commit()

    _mark_job_done(job_id, model_version_id=new_model_version_id)


def _run_training_job(job_id: int):
    try:
        dataset_id, job_type = _start_job(job_id)

        if job_type == "train":
            _run_train_job(job_id, dataset_id)
            return
        if job_type == "change_model_retrain":
            _run_change_model_retrain_job(job_id, dataset_id)
            return

        raise RuntimeError(f"unsupported job type: {job_type}")
    except Exception as exc:
        _mark_job_failed(job_id, str(exc))


def _invalidate_annotator_after_success(job_id: int):
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return
        if job.status != "done":
            return
        invalidate_annotator(job.dataset_id)


def submit_training_job(job_id: int):
    global _executor
    try:
        if _executor is None:
            _executor = ProcessPoolExecutor(max_workers=1)

        future = _executor.submit(_run_training_job, job_id)
        future.add_done_callback(lambda _: _invalidate_annotator_after_success(job_id))
    except Exception as exc:
        _mark_job_failed(job_id, f"failed to submit job: {exc}")
        raise
