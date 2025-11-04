from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import *
import logging
import torch

logging.basicConfig(level=logging.DEBUG)

class EmbeddingModel():
    def __init__(self, device: str = "cpu"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            logging.info("Токенизатор успешно загружен")
        except:
            logging.error("Не удалось загрузить токенизатор")
        try:
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            logging.info("Модель успешно загружена")
        except:
            logging.error("Не удалось загрузить модель")
        device = torch.device(device)
        self.model.to(device)
        logging.info(f"Текущий девайс: {device}")

    def encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings_np = np.array([])
        logging.info("Генерация эмбеддингов...")
        if batch_size >= len(texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            for i in range(0, len(texts), batch_size):
                logging.info(f"Батч {i+1}/{len(texts)/batch_size}")
                batch = texts[i*batch_size:i*batch_size+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0: :]
                embeddings_np = embeddings.detach().cpu().numpy()
        return embeddings_np

    def get_embeddings_dim(self):
        return self.model.config.hidden_size








