from fastapi import FastAPI
from src.embedder import EmbeddingModel
from src.vector_store import VectorStore
import logging
from src.schemas import HealthResponse, SearchRequest, IndexRequest, DocumentInput, IndexResponse, SearchResult, SearchResponse
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
    return HealthResponse(status="healthy", total_documents=len(app.state.vector_store), model_name=app.state.embedder.model_name, embedding_dim=app.state.embedder.get_embeddings_dim())

@app.post("/index")
def index(request: IndexRequest):
    documents = request.documents
    texts = [input.text for input in documents]
    doc_ids = [input.doc_id for input in documents]
    vectors = app.state.embedder.encode(texts)
    app.state.vector_store.add_vectors(vectors, texts, doc_ids)
    response = IndexResponse(indexed_count=len(documents), total_indexed_count=len(app.state.vector_store))
    return response

@app.post("/search")
def search(request: SearchRequest):
    query = request.query
    top_k = request.top_k
    threshold = request.threshold
    if len(app.state.vector_store) == 0:
        response = SearchResponse(query=query, results=[], results_count=0, search_time_ms=0.0, total_documents=0)
        return response
    start_time = time.time()
    vector = app.state.embedder.encode([query])
    results = app.state.vector_store.search(vector, top_k=top_k, threshold=threshold)
    search_time = (time.time() - start_time) * 1000
    results = [SearchResult(doc_id=result['metadata']['doc_id'], text=result['metadata']['text'], score=result['score'], index_position=result['metadata']['index_position']) for result in results]
    response = SearchResponse(query=query, results=results, results_count=len(results), search_time_ms=search_time)
    return response

@app.post("/index/clear")
def clear():
    app.state.vector_store.reset()
    return {"message": "Index cleared"}





