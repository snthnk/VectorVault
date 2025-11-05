from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Секция эмбеддера
    model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="MODEL_NAME")
    batch_size: int = Field(10, ge=1, env="BATCH_SIZE")
    device: str = Field("cpu", env="DEVICE")

    # Секция векторного хранилища
    use_cosine_similarity: bool = Field(True, env="USE_COSINE_SIMILARITY")
    default_top_k: int = Field(5, ge=1, env="DEFAULT_TOP_K")
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0, env="SIMILARITY_THRESHOLD")

    # Секция персистентности
    index_path: Path = Field("data/faiss", env="INDEX_PATH")
    metadata_path: Path = Field("data/metadata", env="METADATA_PATH")
    autosave_interval: int = Field(5, env="AUTOSAVE_INTERVAL")

    # Секция API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

settings = Settings()
