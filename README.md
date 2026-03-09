
##  RAG-Based Multi-Document Question Answering System

A production-style Retrieval-Augmented Generation (RAG) system that allows users to upload multiple PDFs and ask questions grounded strictly in the uploaded documents.

The system uses:

FAISS for vector retrieval

BGE embeddings for semantic search

Cross-Encoder for reranking

Groq LLM (Llama 3.1) for final answer generation

## System Overview

This system enables:

Uploading multiple PDF documents

Automatic chunking and embedding

Persistent vector indexing

Semantic retrieval + reranking

Grounded answer generation

# Architecture
User Query
    ↓
Vector Search (FAISS - top 15 or 20)
    ↓
Cross-Encoder Rerank
    ↓
Top-k Context Selection
    ↓
LLM (Groq Llama 3.1)
    ↓
Final Answer

Upload Flow:

Upload PDF
    ↓
Document Loader
    ↓
Chunking (1000 size, 200 overlap)
    ↓
Embedding (BGE-base)
    ↓
Store in FAISS
# 1️ Why BGE Embeddings?

Model: BAAI/bge-small-en-v1.5 (Small due to decrease the latency)

Strong semantic performance

Better retrieval quality than MiniLM

Good tradeoff between accuracy and latency

## 2 Why Cross-Encoder Reranking? (To increase top 1 and top 3 retreival Quality)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 

Vector similarity alone is not enough.

Cross-Encoder:

Reads query + document together

Provides deeper semantic ranking

Improves top-1 accuracy significantly

## 3 Why FAISS? (To Store vectors)

Fast vector similarity search

Persistent storage

Efficient for medium-scale datasets

Current index: IndexFlatIP (Done using cosine Similarity)

## Why Chunk Size 1000 & Overlap 200? (On this pair we getting good result)

1000 characters gives sufficient context

200 overlap preserves context continuity

Prevents answer fragmentation

##  Embedding Generation Time (PDF Upload Stage)

Previous Time: 4.13 sec

Current Time: 1.23 sec

Optimization Applied:

Enabled GPU acceleration for embedding generation.

Implemented batch processing while encoding document chunks.

Result:
These optimizations reduced the embedding generation time from 4.13 seconds to 1.23 seconds, improving the PDF ingestion pipeline performance by approximately 70%.


# ⚡ Latency Optimization

We optimized the RAG pipeline to significantly reduce query latency using multiple strategies.

---

# 1️⃣ Initial Latency (Before Optimization)

| Component | Latency |
|-----------|--------|
| Embedding | 757.57 ms |
| Retrieval | 63.77 ms |
| Reranking | 261.19 ms |
| Generation | 1356.86 ms |
| **Total Latency** | **2439.44 ms (~2.4 sec)** |

---

# 🚀 Optimizations Applied

## 2️⃣ Embedding Optimization

**Techniques Used**

- Enabled **GPU acceleration** for embedding computation  
- Implemented **batch processing** for encoding document chunks  

**Result**

Embedding latency was significantly reduced.

| Component | Latency |
|-----------|--------|
| Embedding | 42.92 ms |
| Retrieval | 29.23 ms |
| Reranking | 154.08 ms |
| Generation | 722.72 ms |
| **Total Latency** | **948.99 ms (~0.95 sec)** |

✅ **Latency reduced from 2.4s → 0.95s (~61% improvement)**

---

# ⚡ Smart Cache Strategy

We implemented a **semantic cache system** to avoid recomputing responses for similar queries.

## How It Works

1. Every user query is converted into an **embedding vector**
2. The system compares it with **previous query embeddings**
3. **Semantic similarity** is calculated
4. If **similarity ≥ 90%**, the system returns the **cached answer**

This avoids running the full **RAG pipeline** again.

---

## Benefits

- Handles **spelling mistakes**
- Handles **rephrased questions**
- Handles **similar intent queries**
- Avoids unnecessary:
  - Retrieval
  - Reranking
  - Generation

---

# 3️⃣ Cached Query Latency

If a similar query is found in cache:

| Component | Latency |
|-----------|--------|
| Embedding | 31.06 ms |
| Retrieval | 0 ms |
| Reranking | 0 ms |
| Generation | 0 ms |
| **Total Latency** | **31.49 ms** |

✅ **Latency reduced from 2.4s → 31 ms (~98.7% improvement)**

---

# 📊 Final Performance Comparison

| Stage | Total Latency |
|------|---------------|
| Initial System | 2439 ms |
| After Optimization | 948 ms |
| With Semantic Cache | 31 ms |

---

# 🧠 Additional Feature

## Semantic Search

Implemented **semantic similarity matching** for user queries.

The system can detect:

- Spelling mistakes
- Paraphrased questions
- Similar intent queries

If **similarity ≥ 0.90**, the cached response is returned instantly.

---

# ✅ Result

A **faster and more efficient RAG system** using:

- GPU-based embeddings
- Batch encoding
- Semantic search
- Intelligent caching

This significantly reduces **query latency and computation cost** while improving the **overall user experience**.

# 📊 RAG Evaluation Metrics

To evaluate the performance of the retrieval system, we conducted experiments on **20 sample queries** and measured multiple metrics including **Top-K Accuracy, Recall@K, MRR, and Query Latency**.

---

# 🔎 Retrieval Accuracy

| Metric | Value |
|------|------|
| **Top-1 Accuracy** | 75% |
| **Top-3 Accuracy** | 80% |

**Explanation**

- **Top-1 Accuracy** → The correct chunk appears as the **first retrieved result**.
- **Top-3 Accuracy** → The correct chunk appears **within the top 3 retrieved results**.

---

# 📈 Recall Metrics

| Metric | Value |
|------|------|
| **Recall@1** | 80% |
| **Recall@3** | 95% |

**Explanation**

- **Recall@1** → 80% of queries retrieved the correct chunk in the **top 1 result**.
- **Recall@3** → 95% of queries retrieved the correct chunk within the **top 3 results**.

This shows the retriever is able to capture **most relevant chunks within the first few results**.

---

# 🎯 Mean Reciprocal Rank (MRR)

| Metric | Value |
|------|------|
| **MRR@3** | **0.867** |

**Explanation**

MRR measures how early the **correct document appears in the ranking**.

Higher MRR indicates that the system is retrieving **relevant documents at higher ranks**.

---

# ⏱️ Latency Metrics

| Metric | Value |
|------|------|
| **Average Query Latency** | 0.1404 sec |
| **P95 Latency** | 0.1837 sec |

**Explanation**

- **Average Latency** → Average response time per query.
- **P95 Latency** → 95% of queries complete within **0.1837 seconds**, showing stable system performance.

---

# ✅ Summary

- High **Recall@3 (95%)** indicates strong retrieval capability.
- **MRR = 0.867** shows relevant documents are ranked very early.
- **Low latency (~0.14 sec)** ensures fast query responses.

Overall, the system demonstrates **efficient retrieval with strong ranking quality and low latency**, making it suitable for real-time RAG applications.




## 📈 System Behavior as PDFs Grow

As the document size increases, several system components scale accordingly:

- **Chunk Count Increases Linearly**  
  Larger PDFs generate more text chunks during preprocessing.

- **Embedding Time Increases**  
  More chunks require additional embedding computations.

- **FAISS Index Size Grows**  
  Each chunk embedding is stored in the FAISS vector index.

- **Retrieval Latency May Increase**  
  A larger vector index can slightly increase search time depending on the index type.

---

## ⚠️ Potential Production Bottlenecks

As the system scales to larger datasets or higher traffic, the following components may become bottlenecks:

- **Cross-Encoder Reranking**  
  Cross-encoder models are computationally expensive and may increase **GPU utilization** significantly.

- **FAISS Exact Search at Very Large Scale**  
  Exact similarity search can become slower as the vector database grows to millions of embeddings.

- **GPU Memory Exhaustion**  
  Large batch sizes, multiple models, or concurrent requests may exhaust available GPU memory.

- **Document Upload Embedding Latency**  
  Uploading very large PDFs can increase preprocessing and embedding time.

---

## 🧠 Strict Grounding Prompt

To reduce hallucinations, the system uses a **strict grounding prompt strategy**.

The model is instructed to:

- Generate answers **only from the retrieved context**
- **Avoid adding external knowledge**
- Respond with **"I don't know based on the provided context"** if the answer is not present

This ensures:

- Higher **factual accuracy**
- Lower **hallucination rate**
- Better **trustworthiness of generated responses**
Strict grounding prompt to prevent hallucinations
