from api_predict import router as api_router
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()
app.include_router(api_router)


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)