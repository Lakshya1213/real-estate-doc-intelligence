from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


class DataLoader:
    """
    Handles document loading and splitting for multiple file types.
    """

    MAX_FILE_SIZE_MB = 50  # 50MB limit

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_file(self, file_path: str) -> List[Document]:
        """
        Detect file type, validate size, and load document.
        """

        path = Path(file_path)

        # ✅ Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # ✅ Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File size {file_size_mb:.2f}MB exceeds 50MB limit."
            )

        suffix = path.suffix.lower()

        # ✅ Choose correct loader
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))

        elif suffix == ".txt":
            loader = TextLoader(str(path))

        elif suffix == ".csv":
            loader = CSVLoader(str(path))

        elif suffix == ".xlsx":
            loader = UnstructuredExcelLoader(str(path))

        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))

        elif suffix == ".json":
            loader = JSONLoader(
                file_path=str(path),
                jq_schema=".",
                text_content=False
            )

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        documents = loader.load()

        # ✅ Add metadata
        for doc in documents:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_size_mb"] = round(file_size_mb, 2)

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split loaded documents into smaller chunks for embeddings.
        """
        return self.splitter.split_documents(documents)

    def load_and_split(self, file_path: str) -> List[Document]:
        """
        Full pipeline:
        Load → Validate → Split → Return chunks
        """
        documents = self.load_file(file_path)
        chunks = self.split_documents(documents)
        return chunks
