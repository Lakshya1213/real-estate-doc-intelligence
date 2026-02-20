
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

Model: BAAI/bge-base-en-v1.5

Strong semantic performance

Better retrieval quality than MiniLM

Good tradeoff between accuracy and latency

## 2 Why Cross-Encoder Reranking?

Model: cross-encoder/ms-marco-MiniLM-L-6-v2

Vector similarity alone is not enough.

Cross-Encoder:

Reads query + document together

Provides deeper semantic ranking

Improves top-1 accuracy significantly

## 3 Why FAISS?

Fast vector similarity search

Persistent storage

Efficient for medium-scale datasets

Current index: IndexFlatIP (Done using cosine Similarity)

## Why Chunk Size 1000 & Overlap 200?

1000 characters gives sufficient context

200 overlap preserves context continuity

Prevents answer fragmentation

## Performance Evaluation

Evaluation Dataset: 20 domain-specific questions.

# Metrics measured:

# Accuracy (top-3)

# Average Query Latency

# P95 Latency

Results (k =3 )

Accuracy: 0.75
Average Latency: 0.0910 sec 
P95 Latency: 0.1152 ms

After reranking:

Accuracy improves significantly (typically 75–85% depending on dataset).



## System Behavior as PDFs Grow

As document size increases:

Chunk count increases linearly

Embedding time increases

FAISS index grows

Retrieval latency increases


## What Would Break First in Production?

Cross-Encoder reranking (GPU load increases)

FAISS exact search at very large scale

GPU memory exhaustion

Upload embedding latency


## Safety Measures

File size limit (50MB) to prevent:

GPU OOM

Memory overload

Denial-of-service

Metadata normalization to prevent runtime errors

Strict grounding prompt to prevent hallucinations