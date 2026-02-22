from ultralytics import YOLO
import numpy as np

class AutoAnnotator:
    '''модель'''
    def __init__(self, model_path = "yolo11n.pt"):
        self.model = YOLO(model_path)

    '''метод предикт + заданный порог уверенности'''
    def predict(self, image_path: str | np.ndarray , conf: float  = 0.5):
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
