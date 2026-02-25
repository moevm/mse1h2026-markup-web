from api_predict import router as api_router
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
app.include_router(api_router)
app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
app.mount("/img", StaticFiles(directory="frontend/img"), name="img")
app.mount("/pagesContent", StaticFiles(directory="frontend/pagesContent"), name="pagesContent")

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)