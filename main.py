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
import time

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

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANK_MODEL_NAME)

query_embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

RETRIEVAL_K = 6
RERANK_TOP_K = 3

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------
# CACHES
# ---------------------------------------

retrieval_cache = {}
response_cache = {}
query_embedding_cache = {}

# ---------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def print_latency(embed=0, retrieval=0, rerank=0, generation=0, total=0):

    print("\n----- LATENCY BREAKDOWN -----")
    print(f"Embedding  → {embed:.2f} ms")
    print(f"Retrieval  → {retrieval:.2f} ms")
    print(f"Reranking  → {rerank:.2f} ms")
    print(f"Generation → {generation:.2f} ms")
    print(f"Total      → {total:.2f} ms")
    print("-----------------------------\n")


def check_semantic_cache(query_emb):

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

    chunks = embedding_pipeline.chunk_documents(documents)
    embeddings = embedding_pipeline.embed_chunks(chunks)

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

    vector_store.load()

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
        return retrieval_cache[cache_key], 0, 0

    if not vector_store.index:
        return [], 0, 0

    retrieval_start = time.perf_counter()

    results = vector_store.search(query, top_k=RETRIEVAL_K)

    retrieval_time = (time.perf_counter() - retrieval_start) * 1000

    if not results:
        return [], retrieval_time, 0

    rerank_start = time.perf_counter()

    pairs = [[query, doc["text"]] for doc in results]

    scores = reranker.predict(pairs, batch_size=16)

    for doc, score in zip(results, scores):
        doc["rerank_score"] = float(score)

    results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    final_results = results[:top_k]

    rerank_time = (time.perf_counter() - rerank_start) * 1000

    retrieval_cache[cache_key] = final_results

    return final_results, retrieval_time, rerank_time


# ---------------------------------------
# SEARCH ROUTE
# ---------------------------------------

@app.post("/search")
async def search_documents(request: QueryRequest):

    total_start = time.perf_counter()

    query = request.query.strip().lower()

    # -------------------------
    # EMBEDDING TIMER
    # -------------------------

    embed_start = time.perf_counter()

    query_embedding = query_embedder.encode(query)

    embed_time = (time.perf_counter() - embed_start) * 1000

    # -------------------------
    # SEMANTIC CACHE CHECK
    # -------------------------

    similar_query = check_semantic_cache(query_embedding)

    if similar_query and similar_query in response_cache:

        total_time = (time.perf_counter() - total_start) * 1000

        print("[CACHE HIT] Semantic Response")

        print_latency(
            embed_time,
            0,
            0,
            0,
            total_time
        )

        return response_cache[similar_query]

    # -------------------------
    # EXACT CACHE
    # -------------------------

    if query in response_cache:

        total_time = (time.perf_counter() - total_start) * 1000

        print("[CACHE HIT] Exact Response")

        print_latency(
            embed_time,
            0,
            0,
            0,
            total_time
        )

        return response_cache[query]

    if not vector_store.index:
        return {"answer": "Please upload a document first."}

    # -------------------------
    # RETRIEVE + RERANK
    # -------------------------

    results, retrieval_time, rerank_time = retrieve_and_rerank(
        query,
        RERANK_TOP_K
    )

    if not results:
        return {"answer": "No relevant information found."}

    # -------------------------
    # BUILD CONTEXT
    # -------------------------

    context = ""

    for r in results:
        text = r.get("text")[:500]  # limit tokens

        context += f"""
    Source: {r.get('source')}
    Page: {r.get('page')}
    Content:
    {text}
    """

    prompt = f"""
You are a precise document-based AI assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

    # -------------------------
    # GENERATION TIMER
    # -------------------------

    gen_start = time.perf_counter()

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    gen_time = (time.perf_counter() - gen_start) * 1000

    final_answer = response.choices[0].message.content.strip()

    total_time = (time.perf_counter() - total_start) * 1000

    # -------------------------
    # PRINT LATENCY
    # -------------------------

    print_latency(
        embed_time,
        retrieval_time,
        rerank_time,
        gen_time,
        total_time
    )

    result = {
        "question": query,
        "answer": final_answer,
        "sources": results,
        "latency": {
            "embedding_ms": round(embed_time,2),
            "retrieval_ms": round(retrieval_time,2),
            "rerank_ms": round(rerank_time,2),
            "generation_ms": round(gen_time,2),
            "total_ms": round(total_time,2)
        }
    }

    response_cache[query] = result
    query_embedding_cache[query] = query_embedding

    return result