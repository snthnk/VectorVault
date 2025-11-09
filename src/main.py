from fastapi import FastAPI
from embedder import EmbeddingModel
from vector_store import VectorStore
import logging
from schemas import HealthResponse, DocumentInput, IndexResponse, SearchResult, SearchResponse
from typing import List
from core.config import settings
import time

app = FastAPI(title='VectorVault API')

@app.on_event("startup")
def startup():
    try:
        logging.info("Запуск приложения...")
        app.state.embedder = EmbeddingModel()
        app.state.vector_store = VectorStore(app.state.embedder.get_embeddings_dim())
        app.state.vector_store.load()
        logging.info("Приложение успешно запущено")
    except Exception as e:
        logging.error(f"Ошибка при запуске приложения: {e}")

@app.on_event("shutdown")
def shutdown():
    try:
        logging.error("Выключение приложения...")
        app.state.vector_store.save()
    except Exception as e:
        logging.error(f"Ошибка при завершении работы приложения: {e}")

@app.get("/health")
def health():
    return HealthResponse(status=True, total_documents=len(app.state.vector_store), model_name=app.state.embedder.model_name, embedding_dim=app.state.embedder.get_embeddings_dim())

@app.post("/index")
def index(documents: List[DocumentInput]):
    vectors = app.state.embedder.encode(documents)
    texts = [input.text for input in documents]
    app.state.vector_store.add_vectors(vectors, texts)
    response = IndexResponse(indexed_count=len(documents), total_indexed_count=len(app.state.vector_store), message=True)
    return response

@app.post("/search")
def search(query: str, top_k: int, threshold: float):
    if len(app.state.vector_vault) == 0:
        response = SearchResponse(query=query, results=[], results_count=0, search_time_ms=0.0, total_documents=0)
        return response
    start_time = time.time()
    vector = app.state.encode([query])
    results = app.state.vector_store.search(vector, top_k=top_k, threshold=threshold)
    search_time = time.time() - start_time
    results = [SearchResult(doc_id=result['metadata']['doc_id'], text=result['metadata']['text'], score=result['score'], index_position=result['metadata']['index_position']) for result in results]
    response = SearchResponse(query=query, results=results, results_count=len(results), search_time_ms=search_time)
    return response

@app.post("/index/clear")
def clear():
    app.state.vector_store.reset()
    return

@app.delete("/index/{doc_id}")
def delete(doc_id: str):
    app.state.vector_store.delete_doc(doc_id)
    return




