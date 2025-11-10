from pydantic import BaseModel, conlist, Field
from typing import Optional, Dict, List
from core.config import settings
from typing import Any

class DocumentInput(BaseModel):
    '''
    Описывает один документ для индексации
    '''
    text: str = Field(min_length=1)
    doc_id: str = None
    metadata: Optional[Dict[str, Any]]

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
    message: str = "Documents indexed successfully"

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
    status: str
    total_documents: int
    model_name: str
    embedding_dim: int

