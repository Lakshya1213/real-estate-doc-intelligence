from src.vectorstore import FaissVectorStore

# ---------------------------------
# Load existing FAISS index
# ---------------------------------
vector_store = FaissVectorStore("faiss_store")
vector_store.load()

print("\n🔎 Retrieval Testing Mode")
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

    results = vector_store.search(user_query, top_k=k_value)

    if not results:
        print("No results found.\n")
        continue

    print("\nRetrieved Results:")
    for i, r in enumerate(results):
        print(f"\nRank {i+1}")
        print("Source:", r["source"])
        print("Page:", r["page"])
        print("Score:", r["score"])
        print("Text Preview:", r["text"][:300])
    
    print("\n" + "-"*60 + "\n")
