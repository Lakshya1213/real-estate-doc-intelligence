from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder

# ENV & APP SETUP
load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# LOAD RAG COMPONENTS (NO FAISS LOAD HERE)
from src.data_loaders import DataLoader
from src.embeddings import EmbeddingPipeline
from src.vectorstore import FaissVectorStore

embedding_pipeline = EmbeddingPipeline()
vector_store = FaissVectorStore("faiss_store")

# print("[INFO] Loading Cross-Encoder reranker...")
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANK_MODEL_NAME)

# Number of docs retrieved BEFORE reranking
RETRIEVAL_K = 10   # FAISS retrieval
RERANK_TOP_K = 5   # final chunks sent to LLM

# GROQ CLIENT
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# HOME ROUTE
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    print("[INFO] File uploaded:", file.filename)

    # Load document
    loader = DataLoader()
    documents = loader.load_file(file_path)

    # Chunk + embed
    chunks = embedding_pipeline.chunk_documents(documents)
    embeddings = embedding_pipeline.embed_chunks(chunks)

    # Store in FAISS
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

    #  Load FAISS AFTER creation
    vector_store.load()

    print("[INFO] FAISS index created and loaded")

    return {
        "message": "File uploaded, embeddings created, FAISS index ready.",
        "chunks": len(chunks)
    }

# QUERY MODEL

class QueryRequest(BaseModel):
    query: str

def retrieve_and_rerank(query: str, top_k: int):

    # Guard: FAISS must exist
    if not vector_store.index:
        return []

    # Retrieve
    results = vector_store.search(query, top_k=RETRIEVAL_K)

    if not results:
        return []

    #  Prepare rerank pairs
    pairs = [[query, doc["text"]] for doc in results]

    #  Rerank
    scores = reranker.predict(pairs, batch_size=8)

    for doc, score in zip(results, scores):
        doc["rerank_score"] = float(score)

    #  Sort
    results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]

# SEARCH + RAG ROUTE
@app.post("/search")
async def search_documents(request: QueryRequest):

    ## checks index are present or not
    if not vector_store.index:
        return {
            "answer": "Please upload a document first before searching."
        }

    # Retrieve + rerank(For better ouptut)
    results = retrieve_and_rerank(
        query=request.query,
        top_k=RERANK_TOP_K
)

    if not results:
        return {
            "answer": "No relevant information found in uploaded documents."
        }

    # We Want in this fromat context
    context = ""
    for r in results:
        context += f"""
Source: {r.get('source')}
Page: {r.get('page')}
Content:
{r.get('text')}

---
"""

    prompt = f"""
You are a precise document-based AI assistant.

Your task:
Answer the question using ONLY the provided context.

STRICT RULES:
- Do NOT use outside knowledge.
- Do NOT hallucinate or assume.
- If relevant information exists in the context, you MUST answer.
- If the answer is truly absent, say exactly:
  "I could not find this information in the uploaded documents."

NUMERICAL & FACTUAL RULES (VERY IMPORTANT):
- If numbers, percentages, years, quantities, or monetary values appear in the context,
  you MUST quote them EXACTLY as written.
- Even partial numerical information must be reported.
- Do NOT refuse when numerical values are present.

CITATION RULES:
- Always mention the PDF name.
- Always mention the page number.
- Format citations clearly.

ANSWER FORMAT:
  (Source: <PDF name>, Page <page number>)
- Give a direct answer first.
- Then cite the source in this format:

Context:
{context}

Question:
{request.query}

Answer:
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    final_answer = response.choices[0].message.content.strip()

    return {
        "question": request.query,
        "answer": final_answer,
        "sources": results
    }