from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from activate import Session
from helper import get_annotator
from typing import List
from db import (
    Dataset,
    DatasetImage,
    PredictionBatch,
    PredictionBox,
    TrainingJob,
)
from job_runner import submit_training_job
from datetime import datetime, timezone
import json
import os
from PIL import Image

router = APIRouter()


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
    with Session() as session:
        dataset = session.query(Dataset).filter(Dataset.name == dataset_name).first()
        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"датасет: {dataset_name} не найден"
            )
        session.expunge(dataset)
        return dataset


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _read_gt_boxes(dataset_path: str, filename: str) -> list[dict]:
    label_path = os.path.join(
        dataset_path,
        "labels",
        "train",
        os.path.splitext(filename)[0] + ".txt"
    )
    image_path = os.path.join(dataset_path, filename)

    if not os.path.exists(label_path) or not os.path.exists(image_path):
        return []

    with Image.open(image_path) as image:
        img_w, img_h = image.size

    gt_boxes: list[dict] = []
    with open(label_path, "r") as file:
        for raw_line in file:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            gt_boxes.append(
                {
                    "class_id": class_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return gt_boxes


def _match_image_boxes(pred_boxes: list[dict], gt_boxes: list[dict], iou_threshold: float = 0.5) -> dict:
    used_gt: set[int] = set()
    tp = 0
    fp = 0
    matched_ious: list[float] = []

    for pred in pred_boxes:
        best_gt_idx = -1
        best_iou = 0.0
        pred_tuple = (pred["x1"], pred["y1"], pred["x2"], pred["y2"])
        for idx, gt in enumerate(gt_boxes):
            if idx in used_gt or gt["class_id"] != pred["class_id"]:
                continue
            gt_tuple = (gt["x1"], gt["y1"], gt["x2"], gt["y2"])
            iou_value = _iou(pred_tuple, gt_tuple)
            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_idx = idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(used_gt)
    return {"tp": tp, "fp": fp, "fn": fn, "matched_ious": matched_ious}


def _merge_weighted_metric(
    old_value: float | None,
    old_weight: int,
    batch_value: float,
    batch_weight: int,
) -> float:
    if batch_weight <= 0:
        return round(old_value, 4) if old_value is not None else 0.0
    if old_value is None or old_weight <= 0:
        return round(batch_value, 4)
    merged = (old_value * old_weight + batch_value * batch_weight) / (old_weight + batch_weight)
    return round(merged, 4)


def _complete_batch_and_store_metrics(dataset: Dataset, batch_id: int) -> dict:
    with Session() as session:
        batch = session.get(PredictionBatch, batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail=f"batch id={batch_id} не найден")
        if batch.dataset_id != dataset.id:
            raise HTTPException(
                status_code=400,
                detail="batch не принадлежит указанному датасету"
            )
        if batch.status != "active":
            raise HTTPException(
                status_code=409,
                detail=f"batch id={batch_id} уже завершен или недоступен"
            )

        prediction_rows = (
            session.query(PredictionBox)
            .filter(PredictionBox.batch_id == batch_id)
            .all()
        )

        predictions_by_image: dict[str, list[dict]] = {}
        for row in prediction_rows:
            predictions_by_image.setdefault(row.image_filename, []).append(
                {
                    "class_id": row.class_id,
                    "x1": float(row.x1),
                    "y1": float(row.y1),
                    "x2": float(row.x2),
                    "y2": float(row.y2),
                }
            )

        image_filenames: list[str] = []
        if batch.image_filenames_json:
            try:
                raw_filenames = json.loads(batch.image_filenames_json)
                if isinstance(raw_filenames, list):
                    image_filenames = [
                        name for name in raw_filenames if isinstance(name, str) and name
                    ]
            except json.JSONDecodeError:
                image_filenames = []

        if not image_filenames:
            image_filenames = sorted(set(predictions_by_image.keys()))

        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_matched_ious: list[float] = []
        total_gt_boxes = 0

        for filename in image_filenames:
            gt_boxes = _read_gt_boxes(dataset.path, filename)
            pred_boxes = predictions_by_image.get(filename, [])
            image_metrics = _match_image_boxes(pred_boxes, gt_boxes)
            total_gt_boxes += len(gt_boxes)
            total_tp += image_metrics["tp"]
            total_fp += image_metrics["fp"]
            total_fn += image_metrics["fn"]
            all_matched_ious.extend(image_metrics["matched_ious"])

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mean_iou = (
            sum(all_matched_ious) / len(all_matched_ious)
            if all_matched_ious
            else 0.0
        )

        dataset_row = session.get(Dataset, batch.dataset_id)
        if not dataset_row:
            raise HTTPException(status_code=404, detail="датасет для batch не найден")

        batch_images_count = len(image_filenames)
        batch_boxes_count = total_gt_boxes
        old_boxes_total = dataset_row.metrics_boxes_total or 0
        old_images_total = dataset_row.metrics_images_total or 0

        dataset_row.metric_precision = _merge_weighted_metric(
            dataset_row.metric_precision,
            old_boxes_total,
            precision,
            batch_boxes_count,
        )
        dataset_row.metric_recall = _merge_weighted_metric(
            dataset_row.metric_recall,
            old_boxes_total,
            recall,
            batch_boxes_count,
        )
        dataset_row.metric_f1 = _merge_weighted_metric(
            dataset_row.metric_f1,
            old_boxes_total,
            f1,
            batch_boxes_count,
        )
        dataset_row.metric_mean_iou = _merge_weighted_metric(
            dataset_row.metric_mean_iou,
            old_boxes_total,
            mean_iou,
            batch_boxes_count,
        )
        dataset_row.metrics_boxes_total = old_boxes_total + batch_boxes_count
        dataset_row.metrics_images_total = old_images_total + batch_images_count

        session.query(PredictionBox).filter(
            PredictionBox.batch_id == batch_id
        ).delete(synchronize_session=False)

        batch.status = "completed"
        batch.completed_at = datetime.now(timezone.utc)
        session.commit()

        return {
            "batch_id": batch.id,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
            "images_count": batch_images_count,
            "boxes_count": batch_boxes_count,
            "predicted_boxes_count": len(prediction_rows),
            "dataset_metric_precision": dataset_row.metric_precision,
            "dataset_metric_recall": dataset_row.metric_recall,
            "dataset_metric_f1": dataset_row.metric_f1,
            "dataset_metric_mean_iou": dataset_row.metric_mean_iou,
            "dataset_metrics_boxes_total": dataset_row.metrics_boxes_total,
            "dataset_metrics_images_total": dataset_row.metrics_images_total,
        }


@router.post("/api/train/{dataset_name}", status_code=status.HTTP_202_ACCEPTED)
def train(dataset_name: str):
    dataset = get_dataset_by_name(dataset_name)
    images_dir = dataset.path

    if not os.path.exists(images_dir):
        raise HTTPException(
            status_code=404, detail=f"изображения датасета {dataset_name} не найдены"
        )

    with Session() as session:
        session.query(Dataset).filter(Dataset.id == dataset.id).with_for_update().first()

        active_job = (
            session.query(TrainingJob)
            .filter(
                TrainingJob.dataset_id == dataset.id,
                TrainingJob.status.in_(["queued", "running"])
            )
            .first()
        )
        if active_job:
            raise HTTPException(
                status_code=409,
                detail=f"обучение для датасета {dataset_name} уже запущено"
            )

        job = TrainingJob(dataset_id=dataset.id, status="queued", job_type="train")
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    try:
        submit_training_job(job_id)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="не удалось поставить обучение в очередь"
        )

    return {"job_id": job_id, "status": "queued"}


@router.get("/api/train/jobs/{job_id}")
def get_training_job(job_id: int):
    with Session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"job id={job_id} не найден")

        return {
            "id": job.id,
            "dataset_id": job.dataset_id,
            "status": job.status,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }


@router.post("/api/correct/{dataset_name}")
async def correct(
    dataset_name: str,
    labeled_images: List[LabeledImage],
    batch_id: int | None = Query(default=None),
):
    dataset = get_dataset_by_name(dataset_name)
    images_dir = dataset.path

    if not os.path.exists(images_dir):
        raise HTTPException(
            status_code=404, detail=f"изображения датасета {dataset_name} не найдены"
        )

    if batch_id is not None:
        with Session() as session:
            batch = session.get(PredictionBatch, batch_id)
            if not batch:
                raise HTTPException(status_code=404, detail=f"batch id={batch_id} не найден")
            if batch.dataset_id != dataset.id:
                raise HTTPException(
                    status_code=400,
                    detail="batch не принадлежит указанному датасету"
                )
            if batch.status != "active":
                raise HTTPException(
                    status_code=409,
                    detail=f"batch id={batch_id} уже завершен или недоступен"
                )
                
             # Извлекаем ожидаемый список имён файлов батча
            expected_filenames: list[str] = []
            if batch.image_filenames_json:
                try:
                    raw = json.loads(batch.image_filenames_json)
                    if isinstance(raw, list):
                        expected_filenames = [f for f in raw if isinstance(f, str) and f]
                except json.JSONDecodeError:
                    expected_filenames = []

            # Если ожидаемый список пуст (батч без изображений), считаем это ошибкой конфигурации
            if not expected_filenames:
                raise HTTPException(
                    status_code=400,
                    detail="Batch не содержит списка изображений (image_filenames_json пуст)"
                )
                
            # Собираем имена файлов, присланные фронтендом
            submitted_filenames = {item.filename for item in labeled_images}
            expected_set = set(expected_filenames)

            if submitted_filenames != expected_set:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Batch должен быть завершён полностью. "
                        f"Ожидались файлы: {sorted(expected_set)}, "
                        f"получены: {sorted(submitted_filenames)}"
                    )
                )

    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    saved_images: list[str] = []
    for item in labeled_images:
        safe_filename = os.path.basename(item.filename)
        image_path = os.path.join(images_dir, safe_filename)

        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404, detail=f"{safe_filename} не найден в {dataset_name}"
            )

        annotator.save_labels(
            dataset.path, safe_filename, [ann.model_dump() for ann in item.annotations]
        )
        saved_images.append(safe_filename)

    if saved_images:
        with Session() as session:
            existing_rows = {
                row.filename: row
                for row in session.query(DatasetImage).filter(
                    DatasetImage.dataset_id == dataset.id,
                    DatasetImage.filename.in_(saved_images)
                ).all()
            }

            for filename in saved_images:
                image_path = os.path.join(dataset.path, filename)
                label_path = os.path.join(
                    dataset.path,
                    "labels",
                    "train",
                    os.path.splitext(filename)[0] + ".txt"
                )
                row = existing_rows.get(filename)
                if row:
                    row.image_path = image_path
                    row.label_path = label_path
                    row.status_id = 3
                else:
                    session.add(
                        DatasetImage(
                            dataset_id=dataset.id,
                            filename=filename,
                            image_path=image_path,
                            label_path=label_path,
                            status_id=3,
                        )
                    )
            session.commit()

    if batch_id is not None:
        batch_metrics = _complete_batch_and_store_metrics(dataset, batch_id)
        return {
            "status": "ok",
            "dataset": dataset_name,
            "batch": {
                "id": batch_metrics["batch_id"],
                "status": "completed",
            },
            "batch_metrics": {
                "precision": batch_metrics["precision"],
                "recall": batch_metrics["recall"],
                "f1": batch_metrics["f1"],
                "mean_iou": batch_metrics["mean_iou"],
                "images_count": batch_metrics["images_count"],
                "boxes_count": batch_metrics["boxes_count"],
                "predicted_boxes_count": batch_metrics["predicted_boxes_count"],
            },
            "dataset_metrics": {
                "precision": batch_metrics["dataset_metric_precision"],
                "recall": batch_metrics["dataset_metric_recall"],
                "f1": batch_metrics["dataset_metric_f1"],
                "mean_iou": batch_metrics["dataset_metric_mean_iou"],
                "boxes_total": batch_metrics["dataset_metrics_boxes_total"],
                "images_total": batch_metrics["dataset_metrics_images_total"],
            },
        }

    return {"status": "ok", "dataset": dataset_name}


@router.get("/api/datasets/{dataset_name}/images")
async def list_images(dataset_name: str):
    dataset = get_dataset_by_name(dataset_name)
    images_dir = dataset.path
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="датасет не найден")

    files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return {"dataset": dataset_name, "images": files}


@router.get("/api/datasets/{dataset_name}/images/{filename}")
async def get_image(dataset_name: str, filename: str):
    dataset = get_dataset_by_name(dataset_name)
    safe_filename = os.path.basename(filename)
    image_path = os.path.join(dataset.path, safe_filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="файл не найден")
    return FileResponse(image_path)


@router.post("/api/predict/{dataset_name}/{filename}")
async def predict(dataset_name: str, filename: str):
    dataset = get_dataset_by_name(dataset_name)

    annotator = get_annotator(dataset.id)
    if not annotator:
        raise HTTPException(status_code=500, detail="не удалось загрузить модель")

    safe_filename = os.path.basename(filename)
    image_path = os.path.join(dataset.path, safe_filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="файл не найден")

    boxes = annotator.predict(image_path)

    return [
        {
            "id": i,
            "class_id": b["class_id"],
            "label": b["class_name"],
            "conf": b["confidence"],
            "x1": int(b["x1"]),
            "y1": int(b["y1"]),
            "x2": int(b["x2"]),
            "y2": int(b["y2"]),
        }
        for i, b in enumerate(boxes)
    ]