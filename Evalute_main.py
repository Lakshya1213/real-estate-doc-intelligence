import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from src.vectorstore import FaissVectorStore


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_K = 30
SIMILARITY_THRESHOLD = 0.60


# LOAD MODELS

print("[INFO] Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

print("[INFO] Loading cross-encoder reranker...")
reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")


# LOAD VECTOR STORE

vector_store = FaissVectorStore("faiss_store")
vector_store.load()

print("[INFO] FAISS store loaded successfully.\n")


# RETRIEVE + RERANK

def retrieve_and_rerank(query, top_k):

    results = vector_store.search(query, top_k=RETRIEVAL_K)

    if not results:
        return []

    pairs = [[query, doc["text"]] for doc in results]

    rerank_scores = reranker.predict(pairs)

    for doc, score in zip(results, rerank_scores):
        doc["rerank_score"] = float(score)

    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    return results[:top_k]


# DATASET

evaluation_data = [

    {"query": "Why is agriculture important for India's economy?", "expected_answer": "food security"},
    {"query": "Which sectors are increasingly contributing to rural income?", "expected_answer": "livestock"},
    {"query": "What is the role of horticulture in agriculture?", "expected_answer": "high value crops"},
    {"query": "Why is crop diversification important?", "expected_answer": "reduce risk"},
    {"query": "What is the purpose of e-NAM?", "expected_answer": "digital agricultural market"},
    {"query": "What is the objective of the Digital Agriculture Mission?", "expected_answer": "data driven decision"},
    {"query": "How do Farmer Producer Organisations help farmers?", "expected_answer": "collective bargaining"},
    {"query": "Why are quality seeds important?", "expected_answer": "increase productivity"},
    {"query": "Why is irrigation important in agriculture?", "expected_answer": "water availability"},
    {"query": "What environmental factor is affecting agriculture?", "expected_answer": "climate change"},
    {"query": "Why is agricultural productivity important?", "expected_answer": "limited land"},
    {"query": "What does MSP aim to provide to farmers?", "expected_answer": "price support"},
    {"query": "What role do cooperatives play in agriculture?", "expected_answer": "market access"},
    {"query": "Why is mechanisation important in farming?", "expected_answer": "efficiency"},
    {"query": "Why is investment in agricultural research important?", "expected_answer": "improved crop varieties"},
    {"query": "What challenge does fragmented landholding create?", "expected_answer": "low productivity"},
    {"query": "What is a major water-related challenge in agriculture?", "expected_answer": "water scarcity"},
    {"query": "What role does technology play in agriculture?", "expected_answer": "precision farming"},
    {"query": "Why are agricultural markets important for farmers?", "expected_answer": "better prices"},
    {"query": "What is the long-term goal of agricultural reforms?", "expected_answer": "increase farmer income"},
]


# =====================================
# SEMANTIC SIMILARITY CHECK
# =====================================

def semantic_match(expected_answer, results):

    answer_embedding = embedding_model.encode(
        expected_answer,
        convert_to_tensor=True
    )

    max_score = 0

    for doc in results:

        chunk_embedding = embedding_model.encode(
            doc["text"],
            convert_to_tensor=True
        )

        score = util.cos_sim(answer_embedding, chunk_embedding).item()

        max_score = max(max_score, score)

    return max_score >= SIMILARITY_THRESHOLD, max_score


# =====================================
# EVALUATION FUNCTION
# =====================================

def evaluate(evaluation_data, top_k):

    correct = 0
    total = len(evaluation_data)

    query_times = []

    for item in evaluation_data:

        query = item["query"]
        expected_answer = item["expected_answer"]

        print("\n" + "="*70)
        print(f"Query: {query}")

        start = time.perf_counter()

        results = retrieve_and_rerank(query, top_k)

        end = time.perf_counter()

        latency = end - start
        query_times.append(latency)

        found, score = semantic_match(expected_answer, results)

        print(f"Expected Answer: {expected_answer}")

        # ==============================
        # PRINT RETRIEVED RESULTS
        # ==============================

        print("\nTop Retrieved Chunks:")

        if results:
            for i, doc in enumerate(results):

                print(f"\nRank {i+1}")
                print(f"Rerank Score: {doc['rerank_score']:.3f}")

                text_preview = doc["text"][:300]
                print(f"Text: {text_preview}...")

        else:
            print("No results retrieved.")

        print(f"\nSemantic Similarity Score: {score:.3f}")
        print(f"Match: {found}")
        print(f"Query Latency: {latency:.4f} sec")

        if found:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    avg_latency = np.mean(query_times)
    p95_latency = np.percentile(query_times, 95)

    print("\n" + "="*70)
    print(f"Evaluation Results for k = {top_k}")
    print("="*70)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Average Query Latency: {avg_latency:.4f} sec")
    print(f"P95 Latency: {p95_latency:.4f} sec")

    print("="*70)


# =====================================
# RUN EVALUATION
# =====================================

if __name__ == "__main__":

    print("Running evaluation for k = 3")

    evaluate(evaluation_data, 1)