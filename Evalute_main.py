import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.vectorstore import FaissVectorStore


EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" ## This Model is used for arrangeing the in best order

RETRIEVAL_K = 20  # retrieve top 20 before rerank

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

    results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]


# 20 QUESTION DATASET
evaluation_data = [
    {"query": "What percentage of India's workforce is engaged in agriculture?", "expected_page": 226},
    {"query": "What is the average annual growth rate of agriculture in the last five years?", "expected_page": 228},
    {"query": "What was the decadal growth rate from FY16-FY25?", "expected_page": 228},
    {"query": "What was the total foodgrain production in AY 2024-25?", "expected_page": 228},
    {"query": "What was horticulture production in 2024-25?", "expected_page": 229},
    {"query": "What share of GVA does horticulture contribute?", "expected_page": 229},
    {"query": "What is the MSP formula introduced in 2018-19?", "expected_page": 247},
    {"query": "How many crops are covered under MSP?", "expected_page": 247},
    {"query": "How many farmers were insured under PMFBY in 2024-25?", "expected_page": 250},
    {"query": "How many hectares were covered under PMFBY in 2024-25?", "expected_page": 250},
    {"query": "How much has PMFBY disbursed since inception?", "expected_page": 250},
    {"query": "How much did oilseed production increase between 2014-15 and 2024-25?", "expected_page": 234},
    {"query": "What was domestic edible oil availability in 2023-24?", "expected_page": 234},
    {"query": "How much did edible oil import share decline?", "expected_page": 234},
    {"query": "How many Custom Hiring Centres were established?", "expected_page": 243},
    {"query": "How many PACS are being computerised?", "expected_page": 250},
    {"query": "How many FMD vaccinations were administered since 2020?", "expected_page": 243},
    {"query": "How much foreign exchange was saved by ethanol blending?", "expected_page": 234},
    {"query": "How much did livestock GVA increase between FY15 and FY24?", "expected_page": 228},
    {"query": "What was agriculture growth in Q2 FY 2025-26?", "expected_page": 228},
]


# EVALUATION FUNCTION
def evaluate(evaluation_data, top_k):

    correct = 0
    total = len(evaluation_data)
    query_times = []

    for item in evaluation_data:
        query = item["query"]
        expected_page = item["expected_page"]

        print(f"\nQuery: {query}")

        start = time.time()
        results = retrieve_and_rerank(query, top_k)
        end = time.time()

        query_times.append(end - start)

        found = False
        retrieved_pages = []

        for doc in results:
            internal_page = doc["page"]
            printed_page = internal_page 
            retrieved_pages.append(printed_page)

            if printed_page == expected_page:
                found = True
                break

        print(f"Expected page: {expected_page}")
        print(f"Retrieved pages (printed): {retrieved_pages}")

        if found:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    print(f"\n===== Evaluation for k = {top_k} =====")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Average Query Time: {sum(query_times)/total:.4f} sec")
    print("=====================================\n")


# =====================================
# RUN BOTH k=1 and k=3
# =====================================
if __name__ == "__main__":
    # print("Running evaluation for k = 1")
    # # evaluate([evaluation_data[0]], 1)

    # evaluate(evaluation_data, 1)

    print("Running evaluation for k = 3")
    evaluate(evaluation_data, 3)
