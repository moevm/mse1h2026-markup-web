import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import sys
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")
app.mount("/img", StaticFiles(directory=FRONTEND_DIR / "img"), name="img")
app.mount("/pages", StaticFiles(directory=FRONTEND_DIR / "pages"), name="pages")

@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/stats")
async def stats():
    return FileResponse(FRONTEND_DIR / "pages/stats.html")

@app.get("/datasets")
async def datasets():
    return FileResponse(FRONTEND_DIR / "pages/datasets.html")

@app.get("/work")
async def work():
    return FileResponse(FRONTEND_DIR / "pages/work.html")

@app.get("/utils/select-folder")
async def select_folder():
    system = sys.platform
    try:
        if system == "win32":
            result = subprocess.run(
                [
                    "powershell", "-Command",
                    "[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null;"
                    "$f = New-Object System.Windows.Forms.FolderBrowserDialog;"
                    "$f.ShowDialog() | Out-Null;"
                    "$f.SelectedPath"
                ],
                capture_output=True, text=True, timeout=60
            )
            path = result.stdout.strip()

        elif system == "darwin":
            result = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose folder)'],
                capture_output=True, text=True, timeout=60
            )
            path = result.stdout.strip().rstrip("/")

        else:
            try:
                result = subprocess.run(
                    ["zenity", "--file-selection", "--directory"],
                    capture_output=True, text=True, timeout=60
                )
                path = result.stdout.strip()
            except FileNotFoundError:
                result = subprocess.run(
                    ["kdialog", "--getexistingdirectory", "/"],
                    capture_output=True, text=True, timeout=60
                )
                path = result.stdout.strip()

        if not path:
            return JSONResponse({"error": "Папка не выбрана"}, status_code=400)

        return JSONResponse({"path": path})

    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Таймаут ожидания выбора папки"}, status_code=408)
    except FileNotFoundError as e:
        return JSONResponse({"error": f"Диалог недоступен: {e}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5678)
