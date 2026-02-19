from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import time


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_gpu: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Decide device
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = device

        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[INFO] Loaded embedding model: {model_name} on {self.device}")

    # -------------------------------
    # Chunk Documents
    # -------------------------------
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

    # -------------------------------
    # Generate Embeddings + Timing
    # -------------------------------
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]

        print(f"[INFO] Generating embeddings for {len(texts)} chunks on {self.device}...")

        start_time = time.perf_counter()

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device, 
            normalize_embeddings=True
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        print(f"[⏱] Total Embedding Time: {total_time:.4f} seconds")
        print(f"[⚡] Avg Time per Chunk: {total_time / len(texts):.6f} sec")

        return embeddings
