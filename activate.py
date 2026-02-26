from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Base, DatasetStatus
import os
from dotenv import load_dotenv

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
        if session.query(DatasetStatus).count() == 0:
            statuses = [
                DatasetStatus(id=0, name="Just load"),
                DatasetStatus(id=1, name="Done and verificated"),
                DatasetStatus(id=2, name="Need to verify"),
                DatasetStatus(id=3, name="At work"),
            ]
            session.add_all(statuses)
            session.commit()


def get_session():
    session = Session()
    try:
        yield session
    finally:
        session.close()


