import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BACKEND_URL = "http://backend:8000"

app = FastAPI()

app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
app.mount("/img", StaticFiles(directory="frontend/img"), name="img")
app.mount("/pages", StaticFiles(directory="frontend/pages"), name="pages")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")

@app.get("/stats")
async def stats():
    return FileResponse("frontend/pages/stats.html")

@app.get("/work")
async def work():
    return FileResponse("frontend/pages/work.html")

@app.get("/datasets")
async def datasets():
    return FileResponse("frontend/pages/datasets.html")


# Всё остальное (API, upload, train и т.д.) — проксируем на бэкенд
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    url = f"{BACKEND_URL}/{path}"
    params = dict(request.query_params)
    headers = dict(request.headers)
    body = await request.body()

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.request(
            method=request.method,
            url=url,
            params=params,
            headers=headers,
            content=body,
        )

    return StreamingResponse(
        content=iter([response.content]),
        status_code=response.status_code,
        headers=dict(response.headers),
    )


if __name__ == "__main__":
    uvicorn.run("main_frontend:app", host="0.0.0.0", port=80)
