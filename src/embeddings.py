from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch  # for device check


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_gpu: bool = True  # new param
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Decide device
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = device

        # Load model on selected device
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[INFO] Loaded embedding model: {model_name} on {self.device}")

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
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks on {self.device}...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device  # ensures GPU usage
        )

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
