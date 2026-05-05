from ultralytics import YOLO, RTDETR
from PIL import Image
import numpy as np
import yaml
import os
import shutil
import glob

class AutoAnnotator:
    '''модель'''
    def __init__(self, model_path="yolo11n.pt", model_type: str = "YOLO"):
        self.model_type = model_type
        if self.model_type == "RT-DETR":
            self.model = RTDETR(model_path)
        else:
            self.model = YOLO(model_path)

    def predict(self, image_path: str | np.ndarray, conf: float = 0.5):
        '''метод предикт + заданный порог уверенности'''
        results = self.model.predict(source=image_path, conf=conf, verbose=False)
        annotations = []

        '''результаты'''
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                annotations.append({
                    "class_id":   int(box.cls),
                    "class_name": self.model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                })

        return annotations

    def save_labels(self, dataset_path: str, filename: str, annotations: list[dict]):
        '''конвертация координат в ёло формат и сохранение txt формат'''

        # размер картинки
        image_path = os.path.join(dataset_path, filename)
        with Image.open(image_path) as image:
            img_w, img_h = image.size

        # папка для меток
        labels_dir = os.path.join(dataset_path, "labels", "train")
        os.makedirs(labels_dir, exist_ok=True)

        # нормировка и запись
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
        with open(label_path, "w") as f:
            for ann in annotations:
                x_center = ((ann["x1"] + ann["x2"]) / 2) / img_w
                y_center = ((ann["y1"] + ann["y2"]) / 2) / img_h
                width    = (ann["x2"] - ann["x1"]) / img_w
                height   = (ann["y2"] - ann["y1"]) / img_h
                f.write(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def train(self, dataset_path: str, class_names: list[str], epochs: int = 10, batch_size: int = 16, learning_rate: float = 0.001, imgsz: int = 640, optimizer: str = "AdamW", augment: bool = False) -> tuple[str, int]:
        '''дообучение модели на размеченных данных'''

        # создание yaml конфига
        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        yaml_data = {
            "path":  os.path.abspath(dataset_path),
            "train": "images",  # ← Изображения в папке images/
            "val":   "images",  # ← Валидация тоже там
            "nc":    len(class_names),
            "names": class_names,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        # запуск обучения - results содержит путь к весам
        results = self.model.train(
            data=yaml_path, epochs=epochs,
            imgsz=imgsz, batch=batch_size, lr0=learning_rate,
            optimizer=optimizer,
            # Явное управление гиперпараметрами аугментаций
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
        # определяем следующую версию модели
        models_dir = os.path.join(dataset_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        existing = [f for f in os.listdir(models_dir) if f.startswith("model_v") and f.endswith(".pt")]
        next_version = len(existing) + 1

        # берём путь к лучшим весам из результатов обучения
        trained_weights = os.path.join(results.save_dir, "weights", "best.pt")
        model_save_path = os.path.join(models_dir, f"model_v{next_version}.pt")

        if not os.path.exists(trained_weights):
            raise RuntimeError(f"обученные веса не найдены: {trained_weights}")

        shutil.copy(trained_weights, model_save_path)
        if self.model_type == "RT-DETR":
            self.model = RTDETR(model_save_path)
        else:
            self.model = YOLO(model_save_path)
        
        train_dirs = glob.glob('runs/detect/train*')
        for d in train_dirs:
            shutil.rmtree(d, ignore_errors=True)

        return model_save_path, next_version

    def evaluate(self, dataset_path: str, class_names: list[str]):
        """Продуктовая оценка без model.val():
        class-aware one-to-one matching по текущим GT labels vs текущим predict.
        Метод оставляем для совместимости пайплайна.
        """

        labels_dir = os.path.join(dataset_path, "labels", "train")
        if not os.path.isdir(labels_dir):
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "map50": 0.0,      # legacy-ключ
                "map50_95": 0.0,   # legacy-ключ
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
                candidate = os.path.join(dataset_path, f"{stem}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if not image_path:
                continue

            label_path = os.path.join(labels_dir, label_file)
            gt_boxes = self._read_gt_boxes_px(image_path, label_path)
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

            image_metrics = self._match_image_boxes(pred_boxes, gt_boxes)
            total_tp += image_metrics["tp"]
            total_fp += image_metrics["fp"]
            total_fn += image_metrics["fn"]
            matched_ious.extend(image_metrics["matched_ious"])

            # confusion matrix по matched парам
            for gt_idx, pred_idx in image_metrics["matches"]:
                gt_class = gt_boxes[gt_idx]["class_id"]
                pred_class = pred_boxes[pred_idx]["class_id"]
                if 0 <= gt_class < len(class_names) and 0 <= pred_class < len(class_names):
                    confusion_matrix[gt_class][pred_class] += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
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
            # legacy-ключи для совместимости текущего пайплайна
            "map50": round(precision, 4),
            "map50_95": round(mean_iou, 4),
            "mean_iou": round(mean_iou, 4),
            "confusion_matrix": confusion_matrix,
        }

    def _read_gt_boxes_px(self, image_path: str, label_path: str) -> list[dict]:
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

    def _match_image_boxes(
        self,
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
                iou_value = self._iou(pred_tuple, gt_tuple)

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

    @staticmethod
    def _iou(box_a: tuple, box_b: tuple) -> float:
        '''IoU между двумя боксами (x1, y1, x2, y2)'''
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
