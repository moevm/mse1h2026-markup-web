from ultralytics import YOLO
from PIL import Image
import numpy as np
import yaml
import os


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

    def train(self, dataset_name: str, epochs: int = 10):
        '''дообучение модели на размеченных данных'''

        #путь к датасету
        dataset_path = os.path.join("datasets", dataset_name)

        # создание yaml конфига(под весь датасет)
        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        yaml_data = {
            "path":  os.path.abspath(dataset_path),
            "train": "images/train",
            "val":   "images/train",
            "nc":    len(self.model.names), #колличество классов
            "names": list(self.model.names.values()),
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        # запуск обучения
        self.model.train(data=yaml_path, epochs=epochs, imgsz=640)

