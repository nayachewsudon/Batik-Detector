from fastapi import FastAPI, File, UploadFile, Depends
from inference import predict_image
from sqlalchemy.orm import Session
from database import get_db, engine, Base
from models import Classification

app = FastAPI()

#Create tables on startup if they don't exist (before using alembic)
Base.metadata.create_all(bind=engine)

@app.post("/predict/")
async def create_upload_file(file: UploadFile, db: Session = Depends(get_db)):
    contents = await file.read()
    result = predict_image(contents)

    record = Classification(
        filename = file.filename, 
        predicted_class = result["class"], 
        confidence=result["confidence"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return result

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    records = db.query(Classification).order_by(Classification.id.desc()).all()
    return[
        {
            "id": r.id, 
            "filename": r.filename, 
            "predicted_class": r.predicted_class,
            "confidence": r.confidence,
            "created_at": r.created_at,
        }
        for r in records
    ]

@app.get("/health")
async def health(): 
    return {"status": "ok"}