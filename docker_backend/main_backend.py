from api_predict import router as api_router
from api_datasets import router as datasets_router
from activate import create_db_tables
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import subprocess
import sys

app = FastAPI()
app.include_router(api_router)
app.include_router(datasets_router)


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
            result = subprocess.run(["powershell", "-sta", "-Command", ps], capture_output=True, text=True)
            path = result.stdout.strip()
        elif sys.platform == "darwin":
            result = subprocess.run(["osascript", "-e", "POSIX path of (choose folder)"], capture_output=True, text=True)
            path = result.stdout.strip()
        else:
            result = subprocess.run(["zenity", "--file-selection", "--directory"], capture_output=True, text=True)
            path = result.stdout.strip()

        if path:
            return JSONResponse({"path": path})
        else:
            return JSONResponse({"path": None, "error": "cancelled"})
    except Exception as e:
        return JSONResponse({"path": None, "error": str(e)})


if __name__ == "__main__":
    uvicorn.run("main_backend:app", host="0.0.0.0", port=8000)
