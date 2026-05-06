from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Base, DatasetStatus, TrainingJob, ImageStatus, BoundingBoxClass
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

DATABASE_URL = (
    f"postgresql://"
    f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}"
    f"/{os.getenv('DB_NAME')}"
)

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


def create_db_tables():
    Base.metadata.create_all(engine)

    with Session() as session:
        has_changes = False

        if session.query(DatasetStatus).count() == 0:
            statuses = [
                DatasetStatus(id=0, name="Just load"),
                DatasetStatus(id=1, name="Done and verificated"),
                DatasetStatus(id=2, name="Need to verify"),
                DatasetStatus(id=3, name="At work"),
            ]
            session.add_all(statuses)
            has_changes = True

        if session.query(ImageStatus).count() == 0:
            image_statuses = [
                ImageStatus(id=1, name="Не размечено", code="unlabeled"),
                ImageStatus(id=2, name="Размечено", code="labeled"),
                ImageStatus(
                    id=3, name="Готово для обучения", code="ready_for_training"
                ),
                ImageStatus(
                    id=4,
                    name="Авторазмечено (требует проверки)",
                    code="auto_labeled_pending_review",
                ),
                ImageStatus(id=5, name="Размечено окончательно", code="finalized"),
            ]
            session.add_all(image_statuses)
            has_changes = True

        interrupted_jobs = (
            session.query(TrainingJob).filter(TrainingJob.status == "running").all()
        )
        if interrupted_jobs:
            now = datetime.now(timezone.utc)
            for job in interrupted_jobs:
                job.status = "failed"
                job.error = "interrupted by server restart"
                job.finished_at = now
            has_changes = True

        if has_changes:
            session.commit()


def get_session():
    session = Session()
    try:
        yield session
    finally:
        session.close()
