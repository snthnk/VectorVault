import faiss
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import logging
import json
from src.core.config import settings
import shutil

logging.basicConfig(level=logging.DEBUG)

def get_latest_file(directory: Path, extension: str):
    files = [f for f in directory.glob(f"*{extension}")]
    if not files:
        return None

    latest = 0
    res = None
    for file in files:
        stem = file.stem
        date = stem[-15:]
        date = list(date)
        date.pop(8)
        date = "".join(date)
        date = int(date)
        if date > latest:
            res = file
    return res

class VectorStore():
    def __init__(self, embeddings_dim: int, use_cosine_similarity: bool = None):
        if use_cosine_similarity == None:
            use_cosine_similarity = settings.use_cosine_similarity
        logging.info("Инициализация векторного хранилища...")
        if use_cosine_similarity:
            self.index = faiss.IndexFlatIP(embeddings_dim)
        else:
            self.index = faiss.IndexFlatL2(embeddings_dim)
        self.metadata = []
        self.use_cosine_similarity = use_cosine_similarity
        self.last_n = 0

    def __len__(self):
        return len(self.metadata)

    def doc(self, id: int) -> Dict:
        logging.info("Поиск документа по индексу...")
        if len(self.metadata)-1 < id:
            return {}
        return self.metadata[id-1]

    def save(self, index_path: Path = None, metadata_path: Path = None) -> bool:
        if not index_path:
            index_path = settings.index_path
        if not metadata_path:
            metadata_path = settings.metadata_path
        try:
            logging.info("Сохранение FAISS индекса и метаданных...")
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            index_path = index_path / f"faiss_{timestamp}.index"
            metadata_path = metadata_path / f"metadata_{timestamp}.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))
            logging.info("FAISS индекс успешно сохранен")
        except Exception as e:
            logging.error(f"Ошибка при сохранении FAISS индекса: {e}")
            return False
        try:
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=4)
            logging.info("Метаданные успешно сохранены")
        except Exception as e:
            logging.error(f"Ошибка при сохранении метаданных: {e}")
            return False

        return True

    def load(self, index_path: Path = None, metadata_path: Path = None) -> bool:
        if not index_path:
            index_path = settings.index_path
        if not metadata_path:
            metadata_path = settings.metadata_path
        logging.info("Загрузка FAISS индекса и метаданных")
        try:
            if not index_path.exists():
                logging.warning(f"Директория {index_path} для сохранения FAISS индекса не существует")
                return False
            else:
                last_index_path = get_latest_file(index_path, "index")
                index = faiss.read_index(str(last_index_path))
                self.index = index
                logging.info("FAISS индекс упешно загружен")
        except Exception as e:
            logging.error(f"Ошибка при сохранении FAISS индекса: {e}")
        try:
            if not metadata_path.parent.exists():
                logging.warning(f"Директория {metadata_path} для сохранения метаданных не существует")
                return False
            else:
                last_metadata_path = get_latest_file(metadata_path, "json")
                with open(last_metadata_path, "r") as f:
                    metadata = json.load(f)
                self.metadata = metadata
                logging.info("Метаданные успешно загружены")
        except Exception as e:
            logging.error(f"Ошибка при сохранении метаданных: {e}")
        print(f"Размер загруженного индекса {self.index.ntotal}")
        print(f"Количество загруженных документов {len(self.metadata)}")
        print()
        return True

    def reset(self, index_path: Path = None, metadata_path: Path = None) -> bool:
        logging.info("Очистка индекса, метаданных и файлов хранилища...")
        if not index_path:
            index_path = settings.index_path
            metadata_path = settings.metadata_path
        try:
            self.index.reset()
            self.metadata = []
            shutil.rmtree(index_path)
            shutil.rmtree(metadata_path)
            index_path.mkdir(parents=True, exist_ok=True)
            metadata_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Ошибка при очистке индекса, метаданных и файлов хранилища: {e}")
            return False
        logging.info("Индекс, метаданные и файлы хранилища успешно очищены")
        return True

    def add_vectors(self, vectors: np.ndarray, texts: List[str], doc_ids: Optional[List[str]]=None) -> int:
        logging.info("Добавление векторов...")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.use_cosine_similarity:
            faiss.normalize_L2(vectors)
        index_start = self.index.ntotal
        self.index.add(vectors)
        if not doc_ids:
            for i in range(len(texts)):
                self.metadata.append({"doc_id": index_start+i, "text": texts[i], "index_position": i+index_start, "timestamp": datetime.now().isoformat()})
        else:
            for i in range(len(texts)):
                self.metadata.append({"doc_id": doc_ids[i], "text": texts[i], "index_position": i+index_start, "timestamp": datetime.now().isoformat()})
        if self.last_n + len(texts) >= settings.autosave_interval:
            logging.info("Автоматическое сохранение документов...")
            self.save(settings.index_path, settings.metadata_path)
            self.last_n = 0
        else:
            self.last_n += len(texts)
        return vectors.shape[0]

    def search(self, query_vector: np.ndarray, top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        logging.info("Поиск в пространстве векторов...")
        if self.index.ntotal == 0:
            return []
        if self.use_cosine_similarity:
            faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, top_k)
        res = []
        for i, idx in enumerate(indices[0]):
            if (self.use_cosine_similarity and scores[0][i] >= threshold) or (not self.use_cosine_similarity and scores[0][i] <= threshold):
                res.append({"metadata": self.metadata[idx], "score": scores[0][i]})
        return res


