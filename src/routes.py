from fastapi import APIRouter, UploadFile, File
import os

from src.data_loaders import DataLoader

router = APIRouter()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_location = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    loader = DataLoader()
    chunks = loader.load_and_split(file_location)

    return {
        "message": "File uploaded and processed successfully",
        "chunks_created": len(chunks)
    }
