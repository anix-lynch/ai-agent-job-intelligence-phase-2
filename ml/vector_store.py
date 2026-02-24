# Backward compatibility: use features.vector_store
from features.vector_store import VectorStore, RAGJobMatcher
__all__ = ["VectorStore", "RAGJobMatcher"]
