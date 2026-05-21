from ultralytics import YOLO, RTDETR
import numpy as np
import yaml
import os
import shutil
import glob

from helper import read_gt_boxes_px, match_image_boxes, save_auto_accepted_labels


class AutoAnnotator:
    """модель"""

    def __init__(
        self,
        model_path="yolo11n.pt",
        model_type: str = "YOLO",
        device: str | None = None,
    ):
        self.model_type = model_type
        self.device = device
        if self.model_type == "RT-DETR":
            self.model = RTDETR(model_path)
        else:
            self.model = YOLO(model_path)

    def predict(self, image_path: str | np.ndarray, conf: float = 0.25):
        """метод предикт + заданный порог уверенности"""
        results = self.model.predict(
            source=image_path, conf=conf, verbose=False, device=self.device
        )
        annotations = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                annotations.append(
                    {
                        "class_id": int(box.cls),
                        "class_name": self.model.names[int(box.cls)],
                        "confidence": round(float(box.conf), 3),
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }
                )

        return annotations

    def save_labels(self, dataset_path: str, filename: str, annotations: list[dict]):
        """конвертация координат в YOLO txt format"""
        save_auto_accepted_labels(
            dataset_path=dataset_path,
            filename=filename,
            boxes=annotations,
        )

    def train(
        self,
        dataset_path: str,
        class_names: list[str],
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        imgsz: int = 640,
        optimizer: str = "AdamW",
        augment: bool = False,
        save_dir: str | None = None
    ) -> tuple[str, int]:
        """дообучение модели на размеченных данных"""

        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        yaml_data = {
            "path": os.path.abspath(dataset_path),
            "train": "images",
            "val": "images",
            "nc": len(class_names),
            "names": class_names,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f)

        results = self.model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=learning_rate,
            optimizer=optimizer,
            device=self.device,
            hsv_h=0.0 if not augment else 0.015,
            hsv_s=0.0 if not augment else 0.7,
            hsv_v=0.0 if not augment else 0.4,
            degrees=0.0 if not augment else 0.0,
            translate=0.0 if not augment else 0.1,
            scale=0.0 if not augment else 0.5,
            shear=0.0 if not augment else 0.0,
            perspective=0.0,
            flipud=0.0 if not augment else 0.5,
            fliplr=0.0 if not augment else 0.5,
            mosaic=0.0 if not augment else 1.0,
            mixup=0.0 if not augment else 0.0,
        )

        models_dir = os.path.join(dataset_path, "models")
        os.makedirs(models_dir, exist_ok=True)

        existing_versions = []
        for filename in os.listdir(models_dir):
            if not filename.startswith("model_v") or not filename.endswith(".pt"):
                continue

            version_raw = filename.removeprefix("model_v").removesuffix(".pt")
            try:
                existing_versions.append(int(version_raw))
            except ValueError:
                continue

        next_version = max(existing_versions, default=0) + 1

        trained_weights = os.path.join(results.save_dir, "weights", "best.pt")
        model_save_path = os.path.join(models_dir, f"model_v{next_version}.pt")

        if not os.path.exists(trained_weights):
            raise RuntimeError(f"обученные веса не найдены: {trained_weights}")

        shutil.copy(trained_weights, model_save_path)

        if self.model_type == "RT-DETR":
            self.model = RTDETR(model_save_path)
        else:
            self.model = YOLO(model_save_path)

        train_dirs = glob.glob("runs/detect/train*")
        for directory in train_dirs:
            shutil.rmtree(directory, ignore_errors=True)

        return model_save_path, next_version

    def evaluate(self, dataset_path: str, class_names: list[str]):
        """Продуктовая оценка без model.val():
        class-aware one-to-one matching по текущим GT labels vs текущим predict.
        Метод оставляем для совместимости пайплайна.
        """

        labels_dir = os.path.join(dataset_path, "labels")
        if not os.path.isdir(labels_dir):
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
                "mean_iou": 0.0,
                "confusion_matrix": [[0 for _ in class_names] for _ in class_names],
            }

        total_tp = 0
        total_fp = 0
        total_fn = 0
        matched_ious: list[float] = []
        confusion_matrix = [[0 for _ in class_names] for _ in class_names]

        for label_file in os.listdir(labels_dir):
            if not label_file.lower().endswith(".txt"):
                continue

            stem = os.path.splitext(label_file)[0]
            image_path = None

            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(dataset_path, "images", f"{stem}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            if not image_path:
                continue

            label_path = os.path.join(labels_dir, label_file)
            gt_boxes = read_gt_boxes_px(image_path=image_path, label_path=label_path)

            pred_boxes = [
                {
                    "class_id": int(pred["class_id"]),
                    "x1": float(pred["x1"]),
                    "y1": float(pred["y1"]),
                    "x2": float(pred["x2"]),
                    "y2": float(pred["y2"]),
                }
                for pred in self.predict(image_path, conf=0.25)
            ]

            image_metrics = match_image_boxes(pred_boxes, gt_boxes)

            total_tp += image_metrics["tp"]
            total_fp += image_metrics["fp"]
            total_fn += image_metrics["fn"]
            matched_ious.extend(image_metrics["matched_ious"])

            for gt_idx, pred_idx in image_metrics["matches"]:
                gt_class = gt_boxes[gt_idx]["class_id"]
                pred_class = pred_boxes[pred_idx]["class_id"]

                if 0 <= gt_class < len(class_names) and 0 <= pred_class < len(
                    class_names
                ):
                    confusion_matrix[gt_class][pred_class] += 1

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "map50": round(precision, 4),
            "map50_95": round(mean_iou, 4),
            "mean_iou": round(mean_iou, 4),
            "confusion_matrix": confusion_matrix,
        }