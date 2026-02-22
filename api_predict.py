from fastapi import FastAPI, UploadFile, File
from annotator import AutoAnnotator
from typing import List
from PIL import Image
import io

app = FastAPI()
annotator = AutoAnnotator()

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    response = []

    '''цикл для итерации по фоткам если несколько'''
    for file in files:
        # считывание файла
        contents = await file.read() 
        image = Image.open(io.BytesIO(contents))
        # предикт
        annotations = annotator.predict(image)

        response.append({
            "filename": file.filename,
            "annotations": annotations
        })

    return response
