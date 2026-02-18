from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import time


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 900,      
        chunk_overlap: int = 100,
        device: str = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Auto device detection (GPU if available)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        print(f"[INFO] Loaded embedding model: {model_name}")
        print(f"[INFO] Using device: {self.device}")

    # -----------------------------------
    # 1️⃣ Chunk Documents
    # -----------------------------------
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(documents)

        print(f"[INFO] Split {len(documents)} docs → {len(chunks)} chunks.")
        return chunks

    
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        start_time = time.time()

        texts = [chunk.page_content for chunk in chunks]

        texts = [t.strip() for t in texts if len(t.strip()) > 50]

        texts = list(set(texts))

        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        batch_size = 256 if self.device == "cuda" else 64

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        print(f"[INFO] Embedding time: {round(time.time() - start_time, 2)} seconds")

        return embeddings.astype("float32")
