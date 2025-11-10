from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import *
import logging
import torch
from src.core.config import settings

logging.basicConfig(level=logging.DEBUG)

class EmbeddingModel():
    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        if not model_name:
            self.model_name = settings.model_name
        if not device:
            device = settings.device
        if not batch_size:
            batch_size = settings.batch_size
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logging.info("Токенизатор успешно загружен")
        except Exception as e:
            logging.error(f"Не удалось загрузить токенизатор: {e}")
        try:
            self.model = AutoModel.from_pretrained(self.model_name)
            logging.info("Модель успешно загружена")
        except Exception as e:
            logging.error(f"Не удалось загрузить модель {e}")
        device = torch.device(device)
        self.model.to(device)
        logging.info(f"Текущий девайс: {device}")
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        logging.info("Генерация эмбеддингов...")
        embeddings_np = []
        for i in range(0, len(texts), self.batch_size):
            logging.info(f"Батч {i // self.batch_size + 1}/{(len(texts)+self.batch_size-1)//self.batch_size}")
            batch = texts[i:i+self.batch_size]
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








