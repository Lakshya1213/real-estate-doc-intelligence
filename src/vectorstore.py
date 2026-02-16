import os
import faiss
import pickle
import numpy as np


class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

    def store(self, embeddings: np.ndarray, metadata: list):
        """
        Store embeddings and metadata locally using FAISS.
        """

        # Ensure float32 (FAISS requirement)
        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)

        # Add vectors
        index.add(embeddings)

        # Save index
        faiss.write_index(index, os.path.join(self.persist_dir, "faiss.index"))

        # Save metadata
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        print(f"[INFO] Stored {embeddings.shape[0]} vectors locally.")
