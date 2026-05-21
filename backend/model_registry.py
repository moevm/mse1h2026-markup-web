AVAILABLE_MODELS = [
    {
        "id": "yolo11n",
        "weights": "yolo11n.pt",
        "type": "YOLO",
        "size": "Nano",
        "description": "Самая быстрая, наименее точная",
    },
    {
        "id": "yolo11s",
        "weights": "yolo11s.pt",
        "type": "YOLO",
        "size": "Small",
        "description": "Быстрая, менее точная",
    },
    {
        "id": "yolo11m",
        "weights": "yolo11m.pt",
        "type": "YOLO",
        "size": "Medium",
        "description": "Умеренная скорость, средняя точность",
    },
    {
        "id": "yolo11l",
        "weights": "yolo11l.pt",
        "type": "YOLO",
        "size": "Large",
        "description": "Медленная, высокая точность",
    },
    {
        "id": "yolo11x",
        "weights": "yolo11x.pt",
        "type": "YOLO",
        "size": "X-Large",
        "description": "Очень медленная, максимальная точность",
    },
    {
        "id": "yolo26n",
        "weights": "yolo26n.pt",
        "type": "YOLO",
        "size": "Nano",
        "description": "Самая быстрая, наименее точная, актуальная",
    },
    {
        "id": "yolo26s",
        "weights": "yolo26s.pt",
        "type": "YOLO",
        "size": "Small",
        "description": "Быстрая, менее точная, актуальная",
    },
    {
        "id": "yolo26m",
        "weights": "yolo26m.pt",
        "type": "YOLO",
        "size": "Medium",
        "description": "Умеренная скорость, средняя точность, актуальная",
    },
    {
        "id": "yolo26l",
        "weights": "yolo26l.pt",
        "type": "YOLO",
        "size": "Large",
        "description": "Медленная, высокая точность, актуальная",
    },
    {
        "id": "yolo26x",
        "weights": "yolo26x.pt",
        "type": "YOLO",
        "size": "X-Large",
        "description": "Очень медленная, максимальная точность, актуальная",
    },
    {
        "id": "rtdetr-l",
        "weights": "rtdetr-l.pt",
        "type": "RT-DETR",
        "size": "Large",
        "description": "Медленная, высокая точность, хорошо работает на сложных фонах",
    },
    {
        "id": "rtdetr-x",
        "weights": "rtdetr-x.pt",
        "type": "RT-DETR",
        "size": "X-Large",
        "description": "Очень медленная, максимальная точность, хорошо работает на сложных обьектах",
    },
]


def get_all_models() -> list[dict]:
    """вернуть весь список"""
    return AVAILABLE_MODELS


def get_model_by_id(model_id: str) -> dict | None:
    """найти модель по id, вернуть None если нет такой"""
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model
    return None
