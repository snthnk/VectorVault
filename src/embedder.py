from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import *

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

class EmbeddingModel():
    def encode(self, texts: List[str]) -> np.ndarray:
        inputs = tokenizer(texts, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings_np = embeddings.detach().cpu().numpy()
        return embeddings_np








