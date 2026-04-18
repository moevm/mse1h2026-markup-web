from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing

from activate import Session
from db import Dataset, ModelVersion, TrainingJob
from helper import get_annotator, invalidate_annotator
from training import _train_and_save


_executor: ProcessPoolExecutor | None = None
if multiprocessing.current_process().name == "MainProcess":
    _executor = ProcessPoolExecutor(max_workers=1)


def _mark_job_failed(job_id: int, error_message: str):
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return

        job.status = "failed"
        job.error = error_message
        job.finished_at = datetime.now(timezone.utc)
        session.commit()


def _run_training_job(job_id: int):
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return

        if job.status != "queued":
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.finished_at = None
        job.error = None
        session.commit()
        dataset_id = job.dataset_id

    try:
        with Session() as session:
            dataset = session.get(Dataset, dataset_id)
            if not dataset:
                raise RuntimeError(f"dataset with id={dataset_id} not found")
            session.expunge(dataset)

        annotator = get_annotator(dataset_id)
        if not annotator:
            raise RuntimeError("не удалось загрузить модель")

        _, version = _train_and_save(dataset, annotator)

        model_version_id = None
        with Session() as session:
            model_version = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.dataset_id == dataset_id,
                    ModelVersion.version == version
                )
                .first()
            )
            if model_version:
                model_version_id = model_version.id

        with Session() as session:
            job = session.get(TrainingJob, job_id)
            if not job:
                return

            job.status = "done"
            job.finished_at = datetime.now(timezone.utc)
            job.error = None
            if model_version_id is not None:
                job.model_version_id = model_version_id
            session.commit()
    except Exception as exc:
        _mark_job_failed(job_id, str(exc))


def submit_training_job(job_id: int, dataset_id: int):
    global _executor
    try:
        if _executor is None:
            _executor = ProcessPoolExecutor(max_workers=1)

        future = _executor.submit(_run_training_job, job_id)
        future.add_done_callback(lambda _: invalidate_annotator(dataset_id))
    except Exception as exc:
        _mark_job_failed(job_id, f"failed to submit job: {exc}")
        raise