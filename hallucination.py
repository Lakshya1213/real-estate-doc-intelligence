import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.vectorstore import FaissVectorStore


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_K =20 


print("[INFO] Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

print("[INFO] Loading cross-encoder reranker...")
reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")


vector_store = FaissVectorStore("faiss_store")
vector_store.load()

print("[INFO] FAISS store loaded successfully.\n")


def retrieve_and_rerank(query, top_k):

    results = vector_store.search(query, top_k=RETRIEVAL_K)

    if not results:
        return []

    pairs = []

    for doc in results:
        pairs.append([query, doc["text"]])

    rerank_scores = reranker.predict(pairs)

    for doc, score in zip(results, rerank_scores):
        doc["rerank_score"] = float(score)

    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    return results[:top_k]


evaluation_data = [

{
"query": "Why is agriculture important for India's economy?",
"expected_answer": "Agriculture is important because it ensures food security, provides employment, and supports the livelihoods of rural populations."
},

{
"query": "Which sectors are increasingly contributing to rural income?",
"expected_answer": "Livestock, fisheries, and allied agricultural activities are increasingly contributing to rural income."
},

{
"query": "What is the role of horticulture in agriculture?",
"expected_answer": "Horticulture contributes to agriculture through high value crops such as fruits, vegetables, flowers, and plantation crops."
},

{
"query": "Why is crop diversification important?",
"expected_answer": "Crop diversification reduces production risk, improves income stability, and helps farmers adapt to changing climate conditions."
},

{
"query": "What is the purpose of e-NAM?",
"expected_answer": "e-NAM is a digital agricultural market platform that integrates agricultural markets to enable transparent price discovery and better trading."
},

{
"query": "What is the objective of the Digital Agriculture Mission?",
"expected_answer": "The Digital Agriculture Mission aims to enable data driven decision making in agriculture using digital technologies and agricultural databases."
},

{
"query": "How do Farmer Producer Organisations help farmers?",
"expected_answer": "Farmer Producer Organisations help farmers through collective bargaining, better access to markets, and improved input procurement."
},

{
"query": "Why are quality seeds important?",
"expected_answer": "Quality seeds are important because they improve crop productivity, increase yields, and enhance resistance to pests and diseases."
},

{
"query": "Why is irrigation important in agriculture?",
"expected_answer": "Irrigation is important because it ensures reliable water availability for crops and reduces dependence on rainfall."
},

{
"query": "What environmental factor is affecting agriculture?",
"expected_answer": "Climate change is affecting agriculture through changes in rainfall patterns, temperature increases, and extreme weather events."
}

]


def check_recall_semantic(expected_answer, results, k):

    expected_embedding = embedding_model.encode(expected_answer, normalize_embeddings=True)

    for doc in results[:k]:

        text_embedding = embedding_model.encode(doc["text"], normalize_embeddings=True)

        similarity = np.dot(expected_embedding, text_embedding)

        if similarity > 0.70:
            return 1

    return 0


def compute_mrr(expected_answer, results):

    expected_embedding = embedding_model.encode(expected_answer, normalize_embeddings=True)

    for i, doc in enumerate(results):

        text_embedding = embedding_model.encode(doc["text"], normalize_embeddings=True)

        similarity = np.dot(expected_embedding, text_embedding)

        if similarity > 0.68:
            return 1 / (i + 1)

    return 0


def detect_hallucination(expected_answer, results, k):

    expected_embedding = embedding_model.encode(expected_answer, normalize_embeddings=True)

    for doc in results[:k]:

        text_embedding = embedding_model.encode(doc["text"], normalize_embeddings=True)

        similarity = np.dot(expected_embedding, text_embedding)

        if similarity > 0.68:
            return 0

    return 1


def evaluate(evaluation_data, k):

    total = len(evaluation_data)

    recall_hits = 0
    mrr_total = 0
    hallucination_count = 0

    query_times = []

    for item in evaluation_data:

        query = item["query"]
        expected_answer = item["expected_answer"]

        print("\n" + "="*70)
        print("Query:", query)
        print("Expected Answer:", expected_answer)

        start = time.perf_counter()

        results = retrieve_and_rerank(query, k)

        end = time.perf_counter()

        latency = end - start
        query_times.append(latency)

        r = check_recall_semantic(expected_answer, results, k)
        recall_hits += r

        mrr = compute_mrr(expected_answer, results)
        mrr_total += mrr

        hallucination_flag = detect_hallucination(expected_answer, results, k)
        hallucination_count += hallucination_flag

        print("\nTop Retrieved Chunks:")

        if results:

            for i, doc in enumerate(results):

                print("\nRank", i+1)
                print("Rerank Score:", round(doc["rerank_score"], 3))
                print("Text:", doc["text"][:200], "...")

        else:
            print("No results retrieved")

        print("\nRecall@", k, "Match:", r)
        print("MRR contribution:", round(mrr, 3))
        print("Hallucination Flag:", hallucination_flag)
        print("Query Latency:", round(latency, 4), "sec")


    recall_k = recall_hits / total
    mrr_score = mrr_total / total

    hallucination_rate = hallucination_count / total

    avg_latency = np.mean(query_times)
    p95_latency = np.percentile(query_times, 95)

    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    print("Recall@", k, ":", round(recall_k, 3))
    print("MRR:", round(mrr_score, 3))
    print("Hallucination Rate:", round(hallucination_rate, 3))
    print("Average Query Latency:", round(avg_latency, 4), "sec")
    print("P95 Latency:", round(p95_latency, 4), "sec")


if __name__ == "__main__":

    evaluate(evaluation_data, 3)