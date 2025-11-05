from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import *
import logging
import torch

logging.basicConfig(level=logging.DEBUG)

class EmbeddingModel():
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logging.info("Токенизатор успешно загружен")
        except Exception as e:
            logging.error(f"Не удалось загрузить токенизатор: {e}")
        try:
            self.model = AutoModel.from_pretrained(model_name)
            logging.info("Модель успешно загружена")
        except Exception as e:
            logging.error(f"Не удалось загрузить модель {e}")
        device = torch.device(device)
        self.model.to(device)
        logging.info(f"Текущий девайс: {device}")

    def encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        logging.info("Генерация эмбеддингов...")
        embeddings_np = []
        for i in range(0, len(texts), batch_size):
            logging.info(f"Батч {i // batch_size + 1}/{len(texts)+batch_size-1//batch_size}")
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            if not embeddings_np:
                embeddings_np = embeddings.detach().cpu().numpy()
            else:
                embeddings_np = np.vstack((embeddings_np, embeddings.detach().cpu().numpy()))
        return embeddings_np

    def get_embeddings_dim(self):
        return self.model.config.hidden_size








