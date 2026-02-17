from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Static & Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==============================
# Load RAG Components ONCE
# ==============================

from src.data_loaders import DataLoader
from src.embeddings import EmbeddingPipeline
from src.vectorstore import FaissVectorStore

# Load once at startup
embedding_pipeline = EmbeddingPipeline()
vector_store = FaissVectorStore("faiss_store")

# Groq client (loads once)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# Home Route
# ==============================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ==============================
# Upload Route
# ==============================

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
    chunks = embedding_pipeline.chunk_documents(documents)
    embeddings = embedding_pipeline.embed_chunks(chunks)

    # 4️⃣ Store in FAISS
    vector_store.store(
    embeddings.astype("float32"),
    [
        {
            "text": c.page_content,
            "page": c.metadata.get("page", "Unknown"),
            "source": c.metadata.get("source_file", file.filename)
        }
        for c in chunks
    ]
)


    print("[INFO] Embeddings stored successfully")

    return {
        "message": "Embeddings created & stored locally",
        "filename": file.filename,
        "chunks": len(chunks)
    }

# ==============================
# Search + RAG Route
# ==============================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/search")
async def search_documents(request: QueryRequest):

    # 1️⃣ Retrieve relevant chunks
    results = vector_store.search(request.query, request.top_k)

    if not results:
        return {"answer": "No relevant information found."}

    # 2️⃣ Build structured context properly
    context = ""

    for r in results:
        context += f"""
Source: {r.get('source')}
Page: {r.get('page')}
Content:
{r.get('text')}

---
"""

    # 3️⃣ Create improved RAG prompt
    prompt = f"""
You are a helpful document assistant.

Use ONLY the provided context to answer the question.
If the answer is not found in the context, say:
"I could not find this information in the uploaded documents."

When answering:
- Provide a clear answer.
- Mention the PDF name.
- Mention the page number.
- Do not make up information.

Context:
{context}

Question:
{request.query}

Answer:
"""

    # 4️⃣ Call Groq model
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    final_answer = response.choices[0].message.content.strip()

    print(final_answer)

    return {
        "question": request.query,
        "answer": final_answer,
        "sources": results   
    }
