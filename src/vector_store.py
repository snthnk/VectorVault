import faiss
import numpy as np
from typing import List, Optional, Dict
import random
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)

class VectorStore():
    def __init__(self, embeddings_dim: int, use_cosine_similarity: bool):
        logging.info("Инициализация векторного хранилища...")
        if use_cosine_similarity:
            self.index = faiss.IndexFlatIP(embeddings_dim)
        else:
            self.index = faiss.IndexFlatL2(embeddings_dim)
        self.metadata = []
        self.use_cosine_similarity = use_cosine_similarity

    def __len__(self):
        return len(self.metadata)

    def doc(self, id: int) -> Dict:
        logging.info("Поиск документа по индексу...")
        for i in range(len(self.metadata)):
            if self.metadata[i]["doc_id"] == id:
                return self.metadata[i]
        return {}

    def add_vectors(self, vectors: np.ndarray, texts: List[str], doc_ids: Optional[List[str]]=None) -> int:
        logging.info("Добавление векторов...")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.use_cosine_similarity:
            faiss.normalize_L2(vectors)
        index_start = self.index.ntotal
        self.index.add(vectors)
        if not doc_ids:
            for i in range(len(texts)):
                self.metadata.append({"doc_id": random.randint(1, 100000), "text": texts[i], "index_position": i+index_start, "timestamp": datetime.now().isoformat()})
        else:
            for i in range(len(texts)):
                self.metadata.append({"doc_id": doc_ids[i], "text": texts[i], "index_position": i+index_start, "timestamp": datetime.now().isoformat()})
        return vectors.shape[0]

    def search(self, query_vector: np.ndarray, top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        logging.info("Поиск в пространстве векторов...")
        if self.index.ntotal == 0:
            return []
        if self.use_cosine_similarity:
            faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        res = []
        for i, idx in enumerate(indices[0]):
            if distances[0][i] >= threshold:
                res.append({"metadata": self.metadata[idx], "distance": distances[0][i]})
        return res


