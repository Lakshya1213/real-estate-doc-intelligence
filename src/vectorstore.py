import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


# 🔥 Load embedding model once when server starts
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")

    # ---------------- STORE (Append Mode) ----------------
    def store(self, embeddings: np.ndarray, metadata: list):

        embeddings = embeddings.astype("float32")
        dimension = embeddings.shape[1]

        # If index already exists → load and append
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load()

            self.index.add(embeddings)
            self.metadata.extend(metadata)

            print(f"[INFO] Appended {embeddings.shape[0]} vectors.")

        else:
            # First time creation
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            self.metadata = metadata

            print(f"[INFO] Created new index with {embeddings.shape[0]} vectors.")

        # Save updated index
        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Total vectors in index: {self.index.ntotal}")

    # ---------------- LOAD ----------------
    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("FAISS index not found. Upload document first.")

        self.index = faiss.read_index(self.index_path)

        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print("[INFO] FAISS index loaded successfully.")

    # ---------------- SEARCH ----------------
    def search(self, query_text: str, top_k: int = 5):

        if self.index is None:
            self.load()

        # Encode query
        query_embedding = embedding_model.encode([query_text]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({
                    "text": self.metadata[idx].get("text"),
                    "page": self.metadata[idx].get("page", "Unknown"),
                    "source": self.metadata[idx].get("source", "Unknown"),
                    "score": float(dist)
                })

        return results
    
    
   
