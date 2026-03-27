from api_predict import router as api_router
from api_datasets import router as datasets_router
from activate import create_db_tables
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import subprocess
import sys

app = FastAPI()
app.include_router(api_router)
app.include_router(datasets_router)

app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
app.mount("/img", StaticFiles(directory="frontend/img"), name="img")
app.mount("/pages", StaticFiles(directory="frontend/pages"), name="pages")

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")

@app.get("/stats")
async def index():
    return FileResponse("frontend/pages/stats.html")

@app.get("/work")
async def index():
    return FileResponse("frontend/pages/work.html")

@app.get("/datasets")
async def index():
    return FileResponse("frontend/pages/datasets.html")

@app.get("/utils/select-folder")
async def select_folder():
    try:
        if sys.platform == "win32":
            ps = (
                "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
                "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$f.Description = 'Выберите папку'; "
                "$f.ShowNewFolderButton = $true; "
                "$result = $f.ShowDialog((New-Object System.Windows.Forms.Form -Property @{TopMost=$true})); "
                "if ($result -eq 'OK') { Write-Output $f.SelectedPath }"
            )
            result = subprocess.run(
                ["powershell", "-sta", "-Command", ps],
                capture_output=True, text=True
            )
            path = result.stdout.strip()
        elif sys.platform == "darwin":
            result = subprocess.run(
                ["osascript", "-e", "POSIX path of (choose folder)"],
                capture_output=True, text=True
            )
            path = result.stdout.strip()
        else:
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory"],
                capture_output=True, text=True
            )
            path = result.stdout.strip()

        if path:
            return JSONResponse({"path": path})
        else:
            return JSONResponse({"path": None, "error": "cancelled"})

    except Exception as e:
        return JSONResponse({"path": None, "error": str(e)})

# создать если нет 
#create_db_tables()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
