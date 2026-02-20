from src.vectorstore import FaissVectorStore
from sentence_transformers import CrossEncoder


# Load FAISS index
vector_store = FaissVectorStore("faiss_store")
vector_store.load()

# Load reranker model

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval + Reranking function
def retrieve_and_rerank(query, top_k, retrieval_k=20):
    """
    query: user query
    top_k: final results after reranking
    retrieval_k: number of docs retrieved from FAISS
    """

    #  Retrieve from FAISS
    results = vector_store.search(query, top_k=retrieval_k)

    if not results:
        return []

    #  Prepare query-doc pairs
    pairs = [[query, doc["text"]] for doc in results]

    #  Rerank
    rerank_scores = reranker.predict(pairs)

    for doc, score in zip(results, rerank_scores):
        doc["rerank_score"] = float(score)

    # Step 4: Sort by rerank score
    results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Step 5: Return top_k
    return results[:top_k]


# Interactive Testing Mode
print("\n🔎 Retrieval + Reranking Testing Mode")
print("Type 'exit' to stop.\n")

while True:
    user_query = input("Enter query: ")

    if user_query.lower() == "exit":
        print("Exiting testing mode.")
        break

    try:
        k_value = int(input("Enter top_k value: "))
    except ValueError:
        print("Please enter a valid number.\n")
        continue

    results = retrieve_and_rerank(user_query, top_k=k_value)

    if not results:
        print("No results found.\n")
        continue

    print("\nFinal Reranked Results:")
    for i, r in enumerate(results):
        print(f"\nRank {i+1}")
        print("Source:", r.get("source"))
        print("Page:", r.get("page"))
        print("FAISS Score:", r.get("score"))
        print("Rerank Score:", r.get("rerank_score"))
        print("Text Preview:", r["text"][:300])

    print("\n" + "-" * 60 + "\n")