import faiss
import numpy as np
from typing import List, Optional, Dict

class VectorStore():
    def __init__(self, embeddings_dim: int, use_cosine_similarity: bool):
        if use_cosine_similarity:
            self.index = faiss.IndexFlatIP(embeddings_dim)
        else:
            self.index = faiss.IndexFlatL2(embeddings_dim)
        self.metadata = {}
        self.use_cosine_similarity = use_cosine_similarity

    def add_vectors(self, vectors: np.ndarray, texts: List[str], doc_ids: Optional[List[str]]=None) -> List[Dict]:
        if self.use_cosine_similarity:
            normalized = faiss.normalize_L2(vectors)
            self.index.add(normalized)
        else:
            self.index.add(vectors)
        self.metadata = {doc_ids[i]: texts[i] for i in range(len(texts))}
        return vectors.shape[0]

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.use_cosine_similarity:
            normalized = faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        res = []
        for idx in indices:
            res.append(self.metadata[idx])
        return res


