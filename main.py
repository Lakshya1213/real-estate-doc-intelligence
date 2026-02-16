from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


from src.data_loaders import DataLoader
from src.embeddings import EmbeddingPipeline
from src.vectorstore import FaissVectorStore

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # 1️⃣ Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print("[INFO] File saved:", file.filename)

    # 2️⃣ Load document
    loader = DataLoader()
    documents = loader.load_file(file_path)

    # 3️⃣ Chunk + Embed
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(documents)
    embeddings = emb_pipe.embed_chunks(chunks)

    # 4️⃣ Store in FAISS
    store = FaissVectorStore("faiss_store")
    store.store(embeddings.astype("float32"),
                         [{"text": c.page_content} for c in chunks])

    print("[INFO] Embeddings stored successfully")

    return {
        "message": "Embeddings created & stored locally",
        "filename": file.filename,
        "chunks": len(chunks)
    }

