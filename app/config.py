import os
from typing import Optional

class Config:
    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6334))
    COLLECTION_NAME: str = "chat_memory"
    
    # Ollama Configuration
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 10
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    
    # Cache Configuration
    MAX_CACHE_SIZE: int = 100
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", 500))
    
    # Model Configuration
    # Embeddings provider: 'local' (sentence-transformers) or 'ollama'
    EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "ollama")
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # Ollama embeddings model (e.g., 'all-minilm', 'nomic-embed-text')
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
    DEFAULT_LLM_MODEL: str = "Phi4-mini"
    
    # Debug Configuration
    DEBUG_MODE: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Performance
    MAX_CONCURRENT_UPLOADS: int = 3
    STREAM_TIMEOUT: int = 120
    MAX_MESSAGE_LENGTH: int = 5000
    
    # Feature Flags
    ASYNC_EMBEDDING_ENABLED: bool = os.getenv("ASYNC_EMBEDDING_ENABLED", "true").lower() == "true"