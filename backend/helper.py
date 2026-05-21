from db import Dataset, ModelVersion, TrainingConfig, PredictionBatch, PredictionBox
from activate import Session
import os
from PIL import Image
from sqlalchemy import desc
from model_registry import get_model_by_id
import json
from datetime import datetime, timezone
from fastapi import HTTPException

# dataset_id -> экземпляр AutoAnnotator
_annotators = {}


def get_annotator(dataset_id: int):
    from annotator import AutoAnnotator
    """получение экземпляра AutoAnnotator для конкретного датасета.
    если уже есть в кеше - вернёт его.
    если нет - посмотрит в бд есть ли обученная модель,
    если есть - загрузит её веса,
    если нет - возьмёт pretrained базовую модель по архитектуре датасета"""

    # 1. проверяем кеш
    if dataset_id in _annotators:
        return _annotators[dataset_id]

    # 2. лезем в бд
    with Session() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            return None

        architecture = dataset.current_model_architecture

        # Достаем девайс из базы
        config = (
            session.query(TrainingConfig)
            .filter(TrainingConfig.dataset_id == dataset_id)
            .first()
        )
        device = config.device if config else None

        if device is not None and str(device).strip() not in ("cpu", "cuda") and not str(device).startswith("cuda:"):
            device = "cpu"

        # 3. ищем последнюю обученную модель для этого датасета
        last_model = (
            session.query(ModelVersion)
            .filter(
                ModelVersion.dataset_id == dataset_id, ModelVersion.is_active == True
            )
            .order_by(desc(ModelVersion.version))
            .first()
        )

        # 4. если есть обученная — грузим её веса, если нет — берём pretrained
        if last_model and last_model.path:
            model_info = get_model_by_id(last_model.architecture)
            if not model_info:
                return None
            annotator = AutoAnnotator(
                model_path=last_model.path, model_type=model_info["type"], device=device
            )
        else:
            model_info = get_model_by_id(architecture)
            if not model_info:
                return None
            annotator = AutoAnnotator(
                model_path=model_info["weights"],
                model_type=model_info["type"],
                device=device,
            )

        # 5. кладём в кеш
        _annotators[dataset_id] = annotator
        return annotator


def invalidate_annotator(dataset_id: int):
    """сброс кеша для датасета.
    вызывать после дообучения или смены модели,
    чтобы при следующем запросе загрузились новые веса"""

    _annotators.pop(dataset_id, None)

def save_auto_accepted_labels(dataset_path: str, filename: str, boxes: list[dict]) -> None:
    image_path = os.path.join(dataset_path, "images", filename)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"изображение не найдено: {image_path}")

    with Image.open(image_path) as image:
        img_w, img_h = image.size

    labels_dir = os.path.join(dataset_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

    with open(label_path, "w", encoding="utf-8") as file:
        for box in boxes:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])

            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            file.write(
                f"{int(box['class_id'])} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}\n"
            )


def get_missing_gt_labels(dataset_path: str, filenames: list[str]) -> list[str]:
    missing: list[str] = []

    for filename in filenames:
        label_path = os.path.join(
            dataset_path,
            "labels",
            os.path.splitext(filename)[0] + ".txt",
        )
        if not os.path.exists(label_path):
            missing.append(filename)

    return missing


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def read_gt_boxes_px(image_path: str, label_path: str) -> list[dict]:
    if not os.path.exists(label_path) or not os.path.exists(image_path):
        return []

    with Image.open(image_path) as image:
        img_w, img_h = image.size

    gt_boxes: list[dict] = []

    with open(label_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
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


def match_image_boxes(
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    used_gt: set[int] = set()
    tp = 0
    fp = 0
    matched_ious: list[float] = []
    matches: list[tuple[int, int]] = []

    for pred_idx, pred in enumerate(pred_boxes):
        best_gt_idx = -1
        best_iou = 0.0
        pred_tuple = (pred["x1"], pred["y1"], pred["x2"], pred["y2"])

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue

            if gt["class_id"] != pred["class_id"]:
                continue

            gt_tuple = (gt["x1"], gt["y1"], gt["x2"], gt["y2"])
            iou_value = compute_iou(pred_tuple, gt_tuple)

            if iou_value > best_iou:
                best_iou = iou_value
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt_idx)
            tp += 1
            matched_ious.append(best_iou)
            matches.append((best_gt_idx, pred_idx))
        else:
            fp += 1

    fn = len(gt_boxes) - len(used_gt)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_ious": matched_ious,
        "matches": matches,
    }


def merge_weighted_metric(
    old_value: float | None,
    old_weight: int,
    batch_value: float,
    batch_weight: int,
) -> float:
    if batch_weight <= 0:
        return round(old_value, 4) if old_value is not None else 0.0

    if old_value is None or old_weight <= 0:
        return round(batch_value, 4)

    merged = (old_value * old_weight + batch_value * batch_weight) / (
        old_weight + batch_weight
    )
    return round(merged, 4)


def load_batch_filenames(batch: PredictionBatch) -> list[str]:
    if not batch.image_filenames_json:
        return []

    try:
        raw_filenames = json.loads(batch.image_filenames_json)
    except json.JSONDecodeError:
        return []

    if not isinstance(raw_filenames, list):
        return []

    return [name for name in raw_filenames if isinstance(name, str) and name]


def complete_batch_and_update_dataset_metrics(dataset_id: int, batch_id: int) -> dict:
    with Session() as session:
        batch = session.get(PredictionBatch, batch_id)
        if not batch:
            raise HTTPException(status_code=404, detail=f"batch id={batch_id} не найден")

        if batch.dataset_id != dataset_id:
            raise HTTPException(
                status_code=400,
                detail="batch не принадлежит указанному датасету",
            )

        if batch.status != "active":
            raise HTTPException(
                status_code=409,
                detail=f"batch id={batch_id} уже завершен или недоступен",
            )

        dataset = session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="датасет для batch не найден")

        image_filenames = load_batch_filenames(batch)
        if not image_filenames:
            raise HTTPException(
                status_code=400,
                detail="batch не содержит список изображений",
            )

        missing_gt = get_missing_gt_labels(dataset.path, image_filenames)
        if missing_gt:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "batch не может быть завершен: не для всех изображений есть GT labels",
                    "missing": missing_gt,
                },
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

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt_boxes = 0
        all_matched_ious: list[float] = []

        for filename in image_filenames:
            image_path = os.path.join(dataset.path, "images", filename)
            label_path = os.path.join(
                dataset.path,
                "labels",
                os.path.splitext(filename)[0] + ".txt",
            )

            gt_boxes = read_gt_boxes_px(image_path=image_path, label_path=label_path)
            pred_boxes = predictions_by_image.get(filename, [])

            image_metrics = match_image_boxes(pred_boxes, gt_boxes)

            total_gt_boxes += len(gt_boxes)
            total_tp += image_metrics["tp"]
            total_fp += image_metrics["fp"]
            total_fn += image_metrics["fn"]
            all_matched_ious.extend(image_metrics["matched_ious"])

        precision = (
            total_tp / (total_tp + total_fp)
            if (total_tp + total_fp) > 0
            else 0.0
        )
        recall = (
            total_tp / (total_tp + total_fn)
            if (total_tp + total_fn) > 0
            else 0.0
        )
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

        batch_images_count = len(image_filenames)
        batch_boxes_count = total_gt_boxes
        old_boxes_total = dataset.metrics_boxes_total or 0
        old_images_total = dataset.metrics_images_total or 0

        dataset.metric_precision = merge_weighted_metric(
            dataset.metric_precision,
            old_boxes_total,
            precision,
            batch_boxes_count,
        )
        dataset.metric_recall = merge_weighted_metric(
            dataset.metric_recall,
            old_boxes_total,
            recall,
            batch_boxes_count,
        )
        dataset.metric_f1 = merge_weighted_metric(
            dataset.metric_f1,
            old_boxes_total,
            f1,
            batch_boxes_count,
        )
        dataset.metric_mean_iou = merge_weighted_metric(
            dataset.metric_mean_iou,
            old_boxes_total,
            mean_iou,
            batch_boxes_count,
        )

        dataset.metrics_boxes_total = old_boxes_total + batch_boxes_count
        dataset.metrics_images_total = old_images_total + batch_images_count

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
            "dataset_metric_precision": dataset.metric_precision,
            "dataset_metric_recall": dataset.metric_recall,
            "dataset_metric_f1": dataset.metric_f1,
            "dataset_metric_mean_iou": dataset.metric_mean_iou,
            "dataset_metrics_boxes_total": dataset.metrics_boxes_total,
            "dataset_metrics_images_total": dataset.metrics_images_total,
        }