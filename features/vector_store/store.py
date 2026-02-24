"""Vector store: ChromaDB + sentence-transformers for semantic job search."""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import faiss


class VectorStore:
    """Multi-modal vector database for semantic job matching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="job_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        self.dimension = 384
        self.faiss_index = faiss.IndexFlatIP(self.dimension)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids,
        )
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        return results

    def fast_ann_search(self, query: str, top_k: int = 10) -> np.ndarray:
        query_embedding = self.embed_text(query)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1),
            top_k,
        )
        return indices[0], distances[0]


class RAGJobMatcher:
    """RAG pattern: vector search + context for LLM."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def match_jobs(self, resume: str, num_results: int = 5):
        results = self.vector_store.semantic_search(resume, top_k=num_results)
        return results

    def _build_context(self, results) -> str:
        return "\n".join([doc for doc in results["documents"][0]])
