from api_predict import router as api_router
from api_datasets import router as datasets_router
from activate import create_db_tables
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5678", "http://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router)
app.include_router(api_router)

create_db_tables()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
