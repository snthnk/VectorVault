from pydantic import BaseModel, conlist
from typing import Optional, Dict, List
from core.config import settings

class DocumentInput(BaseModel):
    '''
    Описывает один документ для индексации
    '''
    text: str
    doc_id: str = len()
    metadata: Optional[Dict[str, any]]

class IndexRequest(BaseModel):
    '''
    Запрос на индексацию списка документов
    '''
    documents: conlist(DocumentInput, min_length=1)

class SearchRequest(BaseModel):
    '''
    Запрос на поиск
    '''
    query: str
    top_k: int = settings.default_top_k
    threshold: float = settings.similarity_threshold

class IndexResponse(BaseModel):
    '''
    Ответ на запрос на индексацию
    '''
    indexed_count: int
    total_indexed_count: int
    message: bool

class SearchResult(BaseModel):
    '''
    Результат поиска
    '''
    doc_id: str
    text: str
    score: float
    index_position: int

class SearchResponse(BaseModel):
    '''
    Ответ на запрос на поиск
    '''
    query: str
    results: List[SearchResult]
    results_count: int
    search_time_ms: float
    total_documents: int

class HealthResponse(BaseModel):
    '''
    Ответ health check эндпоинта
    '''
    status: bool
    total_documents: int
    model_name: str
    embedding_dim: int

