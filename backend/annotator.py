from ultralytics import YOLO
from PIL import Image
import numpy as np
import yaml
import os
import shutil

class AutoAnnotator:
    '''модель'''
    def __init__(self, model_path="yolo11n.pt"):
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

    def save_labels(self, dataset_name: str, filename: str, annotations: list[dict]):
        '''конвертация координат в ёло формат и сохранение txt формат'''

        # размер картинки
        image_path = os.path.join("datasets", dataset_name, "images", "train", filename)
        image = Image.open(image_path)
        img_w, img_h = image.size

        # папка для меток
        labels_dir = os.path.join("datasets", dataset_name, "labels", "train")
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

    def train(self, dataset_name: str, epochs: int = 10, batch_size: int = 16, learning_rate: float = 0.001, imgsz: int = 640, optimizer: str = "AdamW") -> tuple[str, int]:
        '''дообучение модели на размеченных данных'''

        dataset_path = os.path.join("datasets", dataset_name)

        # создание yaml конфига
        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        yaml_data = {
            "path":  os.path.abspath(dataset_path),
            "train": "images/train",
            "val":   "images/train",
            "nc":    len(self.model.names),
            "names": list(self.model.names.values()),
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        # запуск обучения - results содержит путь к весам
        results = self.model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch_size, lr0=learning_rate, optimizer=optimizer)

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
        self.model = YOLO(model_save_path)

        return model_save_path, next_version

    def evaluate(self, dataset_name: str): 
        '''оценка модели на валидационном наборе'''
        dataset_path = os.path.join("datasets", dataset_name)

        # создание yaml конфига
        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        yaml_data = {
            "path":  os.path.abspath(dataset_path),
            "train": "images/train",
            "val":   "images/train",
            "nc":    len(self.model.names),
            "names": list(self.model.names.values()),
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        # запуск валидации  results содержит путь к весам
        results = self.model.val(data=yaml_path)
        # извлекаем метрики
        precision = float(results.box.mp)    # mean precision по всем классам
        recall = float(results.box.mr)       # mean recall по всем классам
        map50 = float(results.box.map50)     # mAP@50
        map50_95 = float(results.box.map)    # mAP@50:95

        # F1 считаем вручную
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        # средний IoU  считаем реальный IoU по предсказаниям vs ground truth
        mean_iou = self._compute_mean_iou(dataset_name)

        # confusion matrix в JSON
        confusion_matrix = results.confusion_matrix.matrix.tolist()

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "mean_iou": round(mean_iou, 4),
            "confusion_matrix": confusion_matrix
        }

    def _compute_mean_iou(self, dataset_name: str) -> float:
        '''вычисление реального mean IoU: предсказания vs ground truth метки'''
        images_dir = os.path.join("datasets", dataset_name, "images", "train")
        labels_dir = os.path.join("datasets", dataset_name, "labels", "train")

        if not os.path.exists(labels_dir):
            return 0.0

        all_ious = []
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue

            # читаем ground truth в формате YOLO (нормализованные координаты)
            label_path = os.path.join(labels_dir, label_file)
            gt_boxes = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    gt_boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))

            if not gt_boxes:
                continue

            # находим соответствующее изображение
            image_name = None
            for ext in ('.jpg', '.jpeg', '.png'):
                candidate = os.path.splitext(label_file)[0] + ext
                if os.path.exists(os.path.join(images_dir, candidate)):
                    image_name = candidate
                    break
            if not image_name:
                continue

            image_path = os.path.join(images_dir, image_name)
            img = Image.open(image_path)
            img_w, img_h = img.size

            # предсказания модели
            preds = self.predict(image_path, conf=0.25)
            if not preds:
                continue

            # нормализуем предсказания
            pred_boxes = [
                (p["x1"] / img_w, p["y1"] / img_h, p["x2"] / img_w, p["y2"] / img_h)
                for p in preds
            ]

            # для каждого GT бокса находим лучший pred по IoU
            for gt in gt_boxes:
                best_iou = 0.0
                for pred in pred_boxes:
                    iou = self._iou(gt, pred)
                    if iou > best_iou:
                        best_iou = iou
                all_ious.append(best_iou)

        return sum(all_ious) / len(all_ious) if all_ious else 0.0

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
