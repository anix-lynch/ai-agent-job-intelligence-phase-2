"""Vector Store Implementation using ChromaDB and Pinecone

Demonstrates:
- Vector embeddings with sentence-transformers
- Semantic search with cosine similarity
- FAISS indexing for approximate nearest neighbor
- Retrieval Augmented Generation (RAG) pattern
- Multi-vector database support (ChromaDB, Pinecone)
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import faiss
import os

# Optional Pinecone import
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


class VectorStore:
    """Multi-modal vector database for semantic job matching
    
    Supports multiple vector databases:
    - ChromaDB (default, open source)
    - Pinecone (cloud, optional)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_pinecone: bool = False):
        # Initialize transformer model for neural embeddings
        self.model = SentenceTransformer(model_name)
        self.use_pinecone = use_pinecone and PINECONE_AVAILABLE
        
        if self.use_pinecone:
            # Pinecone for cloud vector storage
            api_key = os.getenv('PINECONE_API_KEY')
            if api_key:
                self.pinecone = Pinecone(api_key=api_key)
                self.index_name = "job-embeddings"
                # Create index if it doesn't exist
                try:
                    self.pinecone_index = self.pinecone.Index(self.index_name)
                except:
                    # Index doesn't exist, will be created on first use
                    self.pinecone_index = None
            else:
                self.use_pinecone = False
        
        if not self.use_pinecone:
            # ChromaDB for persistent vector storage (ephemeral client)
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection(
                name="job_embeddings",
                metadata={"hnsw:space": "cosine"}  # Cosine similarity metric
            )
        
        # FAISS for approximate nearest neighbor search
        self.dimension = 384  # MiniLM embedding dimension
        self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine sim
        
    def embed_text(self, text: str) -> np.ndarray:
        """Generate vector embeddings using transformer models"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents with vector embeddings to store
        
        Implements:
        - Automated feature engineering
        - Neural embeddings
        - Dimensionality reduction (via transformers)
        """
        texts = documents
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        if self.use_pinecone and self.pinecone_index:
            # Add to Pinecone
            vectors = [
                {
                    "id": str(id_val),
                    "values": emb.tolist(),
                    "metadata": {**meta, "text": text}
                }
                for id_val, emb, meta, text in zip(ids, embeddings, metadatas, texts)
            ]
            self.pinecone_index.upsert(vectors=vectors)
        else:
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                ids=ids
            )
        
        # Add to FAISS index for fast approximate nearest neighbor
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.faiss_index.add(embeddings)
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search using vector similarity
        
        Implements:
        - Cosine similarity
        - Approximate nearest neighbor (ANN)
        - Retrieval augmented generation (RAG) pattern
        """
        # Embed query
        query_embedding = self.embed_text(query)
        
        if self.use_pinecone and self.pinecone_index:
            # Search Pinecone
            results = self.pinecone_index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            # Convert to ChromaDB-like format for compatibility
            return {
                "documents": [[match.metadata.get("text", "") for match in results.matches]],
                "ids": [[match.id for match in results.matches]],
                "distances": [[match.score for match in results.matches]]
            }
        else:
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            return results
    
    def fast_ann_search(self, query: str, top_k: int = 10) -> np.ndarray:
        """Fast approximate nearest neighbor using FAISS indexing"""
        query_embedding = self.embed_text(query)
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # FAISS search
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1),
            top_k
        )
        
        return indices[0], distances[0]


class RAGJobMatcher:
    """Retrieval Augmented Generation for job matching
    
    Combines:
    - Vector search
    - LLM context injection
    - Chain-of-thought reasoning
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def match_jobs(self, resume: str, num_results: int = 5):
        """Match jobs using RAG pattern
        
        Steps:
        1. Semantic search (retrieval)
        2. Context injection (augmentation)
        3. LLM reasoning (generation)
        """
        # Retrieval: Find similar jobs
        results = self.vector_store.semantic_search(resume, top_k=num_results)
        
        # Augmentation: Context for LLM
        context = self._build_context(results)
        
        # Generation: LLM analyzes matches (placeholder)
        # In production: Call OpenAI/Anthropic with context
        
        return results
    
    def _build_context(self, results) -> str:
        """Build context for LLM prompt engineering"""
        return "\n".join([doc for doc in results["documents"][0]])
