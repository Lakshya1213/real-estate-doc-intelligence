import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.vectorstore import FaissVectorStore


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_K = 30


# ==============================
# LOAD MODELS
# ==============================

print("[INFO] Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

print("[INFO] Loading cross-encoder reranker...")
reranker = CrossEncoder(RERANK_MODEL_NAME, device="cuda")


# ==============================
# LOAD VECTOR STORE
# ==============================

vector_store = FaissVectorStore("faiss_store")
vector_store.load()

print("[INFO] FAISS store loaded successfully.\n")


# ==============================
# RETRIEVE + RERANK
# ==============================

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


# =====================================
# DATASET
# =====================================

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
},

{
"query": "Why is agricultural productivity important?",
"expected_answer": "Agricultural productivity is important because land resources are limited and increasing productivity helps meet growing food demand."
},

{
"query": "What does MSP aim to provide to farmers?",
"expected_answer": "Minimum Support Price provides price support and income stability to farmers by guaranteeing a minimum selling price for crops."
},

{
"query": "What role do cooperatives play in agriculture?",
"expected_answer": "Agricultural cooperatives help farmers by improving market access, providing credit, and supporting collective marketing."
},

{
"query": "Why is mechanisation important in farming?",
"expected_answer": "Mechanisation improves farming efficiency, reduces labour requirements, and increases agricultural productivity."
},

{
"query": "Why is investment in agricultural research important?",
"expected_answer": "Investment in agricultural research helps develop improved crop varieties, better farming practices, and higher productivity technologies."
},

{
"query": "What challenge does fragmented landholding create?",
"expected_answer": "Fragmented landholding leads to low productivity and inefficient use of agricultural resources."
},

{
"query": "What is a major water-related challenge in agriculture?",
"expected_answer": "Water scarcity is a major challenge in agriculture due to overuse of groundwater and irregular rainfall."
},

{
"query": "What role does technology play in agriculture?",
"expected_answer": "Technology enables precision farming, improved crop monitoring, and better decision making for farmers."
},

{
"query": "Why are agricultural markets important for farmers?",
"expected_answer": "Agricultural markets help farmers obtain better prices and improve income through efficient marketing systems."
},

{
"query": "What is the long-term goal of agricultural reforms?",
"expected_answer": "The long term goal of agricultural reforms is to increase farmer income and improve the sustainability of the agricultural sector."
}

]

# =====================================
# RECALL CHECK FUNCTION
# =====================================

def check_recall_semantic(expected_answer, results, k):

    expected_embedding = embedding_model.encode(expected_answer, normalize_embeddings=True)

    for doc in results[:k]:

        text_embedding = embedding_model.encode(doc["text"], normalize_embeddings=True)

        similarity = np.dot(expected_embedding, text_embedding)

        if similarity > 0.68:   # threshold
            return 1

    return 0


# =====================================
# EVALUATION FUNCTION
# =====================================

def evaluate(evaluation_data, k):

    total = len(evaluation_data)
    recall_hits = 0
    query_times = []

    for item in evaluation_data:

        query = item["query"]
        expected_answer = item["expected_answer"]

        print("\n" + "="*70)
        print(f"Query: {query}")
        print(f"Expected Answer: {expected_answer}")

        start = time.perf_counter()

        results = retrieve_and_rerank(query, k)

        end = time.perf_counter()

        latency = end - start
        query_times.append(latency)

        r = check_recall_semantic(expected_answer, results, k)

        recall_hits += r

        print("\nTop Retrieved Chunks:")

        if results:
            for i, doc in enumerate(results):

                print(f"\nRank {i+1}")
                print(f"Rerank Score: {doc['rerank_score']:.3f}")
                print(f"Text: {doc['text'][:200]}...")

        else:
            print("No results retrieved")

        print(f"\nRecall@{k} Match: {r}")
        print(f"Query Latency: {latency:.4f} sec")

    recall_k = recall_hits / total

    avg_latency = np.mean(query_times)
    p95_latency = np.percentile(query_times, 95)

    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    print(f"Recall@{k}: {recall_k:.2f}")
    print(f"Average Query Latency: {avg_latency:.4f} sec")
    print(f"P95 Latency: {p95_latency:.4f} sec")
# =====================================
# RUN
# =====================================

if __name__ == "__main__":

    evaluate(evaluation_data,1)