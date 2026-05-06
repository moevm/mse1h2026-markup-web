from db import Dataset, ModelVersion, TrainingConfig
from activate import Session
from annotator import AutoAnnotator
from typing import Optional
from sqlalchemy import desc
from model_registry import get_model_by_id

# dataset_id -> экземпляр AutoAnnotator
_annotators: dict[int, AutoAnnotator] = {}


def get_annotator(dataset_id: int) -> Optional[AutoAnnotator]:
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
