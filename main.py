from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer

# ENV & APP SETUP
load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# LOAD RAG COMPONENTS
from src.data_loaders import DataLoader
from src.embeddings import EmbeddingPipeline
from src.vectorstore import FaissVectorStore

embedding_pipeline = EmbeddingPipeline()
vector_store = FaissVectorStore("faiss_store")

# ---------------------------------------
# MODELS
# ---------------------------------------

# RERANK MODEL
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANK_MODEL_NAME)

# QUERY EMBEDDING MODEL (for semantic cache)
query_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieval settings
RETRIEVAL_K = 10
RERANK_TOP_K = 5

# GROQ CLIENT
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------
# CACHES
# ---------------------------------------

# Retrieval cache
retrieval_cache = {}

# Final response cache
response_cache = {}

# Query embedding cache (for semantic similarity)
query_embedding_cache = {}

# ---------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def check_semantic_cache(query):

    query_emb = query_embedder.encode(query)

    for cached_query, cached_emb in query_embedding_cache.items():

        sim = cosine_similarity(query_emb, cached_emb)

        if sim > 0.90:
            print(f"[SEMANTIC CACHE HIT] Similar to: {cached_query}")
            return cached_query

    return None


# ---------------------------------------
# HOME ROUTE
# ---------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------
# FILE UPLOAD
# ---------------------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    print("[INFO] File uploaded:", file.filename)

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

    # Load FAISS index
    vector_store.load()

    # Clear caches after new document
    retrieval_cache.clear()
    response_cache.clear()
    query_embedding_cache.clear()

    print("[INFO] FAISS index created and loaded")
    print("[INFO] Cache cleared after upload")

    return {
        "message": "File uploaded, embeddings created, FAISS index ready.",
        "chunks": len(chunks)
    }


# ---------------------------------------
# QUERY MODEL
# ---------------------------------------

class QueryRequest(BaseModel):
    query: str


# ---------------------------------------
# RETRIEVE + RERANK
# ---------------------------------------

def retrieve_and_rerank(query: str, top_k: int):

    cache_key = f"{query}_{top_k}"

    if cache_key in retrieval_cache:
        print("[CACHE HIT] Retrieval")
        return retrieval_cache[cache_key]

    if not vector_store.index:
        return []

    # Retrieve from FAISS
    results = vector_store.search(query, top_k=RETRIEVAL_K)

    if not results:
        return []

    # Prepare rerank pairs
    pairs = [[query, doc["text"]] for doc in results]

    scores = reranker.predict(pairs, batch_size=8)

    for doc, score in zip(results, scores):
        doc["rerank_score"] = float(score)

    results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    final_results = results[:top_k]

    retrieval_cache[cache_key] = final_results

    return final_results


# ---------------------------------------
# SEARCH + RAG ROUTE
# ---------------------------------------

@app.post("/search")
async def search_documents(request: QueryRequest):

    query = request.query.strip().lower()

    # -------------------------
    # SEMANTIC CACHE CHECK
    # -------------------------

    similar_query = check_semantic_cache(query)

    if similar_query and similar_query in response_cache:
        print("[CACHE HIT] Semantic Response")
        return response_cache[similar_query]

    # -------------------------
    # NORMAL CACHE CHECK
    # -------------------------

    if query in response_cache:
        print("[CACHE HIT] Exact Response")
        return response_cache[query]

    if not vector_store.index:
        return {
            "answer": "Please upload a document first before searching."
        }

    # Retrieve + rerank
    results = retrieve_and_rerank(
        query=query,
        top_k=RERANK_TOP_K
    )

    if not results:
        return {
            "answer": "No relevant information found in uploaded documents."
        }

    # -------------------------
    # BUILD CONTEXT
    # -------------------------

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

CITATION RULES:
- Always mention the PDF name.
- Always mention the page number.

ANSWER FORMAT:
(Source: <PDF name>, Page <page number>)
Direct answer first, then citation.

Context:
{context}

Question:
{query}

Answer:
"""

    # -------------------------
    # LLM GENERATION
    # -------------------------

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    final_answer = response.choices[0].message.content.strip()

    result = {
        "question": query,
        "answer": final_answer,
        "sources": results
    }

    # -------------------------
    # SAVE CACHE
    # -------------------------

    response_cache[query] = result

    query_embedding_cache[query] = query_embedder.encode(query)

    return result