from fastapi import FastAPI, File, UploadFile
from inference import predict_image

app = FastAPI()

@app.post("/predict/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    return predict_image(contents)

@app.get("/health")
async def health(): 
    return {"status": "ok"}