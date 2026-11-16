"""Configuration settings for FastAPI application."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    PDF_STORAGE_DIR: str = "data/pdfs/uploads"
    VECTOR_DB_DIR: str = "data/vectors"

    # Database
    DATABASE_URL: str = "sqlite:///./data/api.db"

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    DEFAULT_CHAT_MODEL: str = "phi4-mini:3.8b"

    # Streaming chat provider and multi-mode architecture (Stage 4)
    # RAG_MODE: "normal" (Ollama -> Groq), "presentation" (Groq -> Ollama), "offline" (Ollama only)
    CHAT_PROVIDER: str = "ollama"
    RAG_MODE: str = "normal"
    OLLAMA_FALLBACK_TIMEOUT: float = 60.0  # seconds before falling back to Groq in normal mode

    # Groq Cloud API settings (Stage 4)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_API_BASE: str = "https://api.groq.com/openai/v1"
    GROQ_TIMEOUT_SECONDS: float = 30.0
    GROQ_MAX_TOKENS: int = 512
    GROQ_TEMPERATURE: float = 0.1

    # LLM generation (tuned for 8GB CPU-only machines)
    NUM_CTX: int = 2048  # context window; raise to 3072 if context truncates
    NUM_PREDICT: int = 512  # answer token cap keeps worst-case latency bounded
    OLLAMA_KEEP_ALIVE: str = "30m"  # avoid model reload between questions
    LLM_TEMPERATURE: float = 0.1

    # Server
    RELOAD: bool = False  # dev file-watcher (disable for presentations)

    # RAG chunking (Phase 1: configurable, balanced defaults)
    # 1500 chars ≈ 300-400 tokens: better precision than 7500 while
    # preserving paragraph-level context. Overlap 200 preserves sentences.
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200

    # RAG retrieval (Phase 2: hybrid + MMR tuning)
    RETRIEVAL_K: int = 5  # chunks per query per PDF (was 3)
    RETRIEVAL_FETCH_K: int = 20  # candidate pool for MMR
    RETRIEVAL_MAX_CHUNKS: int = 5  # cap after merge/dedup (context budget)
    ENABLE_HYBRID_SEARCH: bool = False  # BM25 full-collection scan; opt-in only
    HYBRID_VECTOR_WEIGHT: float = 0.7  # 70% semantic, 30% keyword
    USE_MMR: bool = True  # maximal marginal relevance for diversity
    USE_MULTI_QUERY: bool = False  # extra LLM call per query; opt-in only

    # OCR fallback (Phase 1)
    ENABLE_OCR_FALLBACK: bool = True
    OCR_MIN_CHARS: int = 50  # below this -> treat as scanned

    # Logging
    LOG_LEVEL: str = "INFO"


settings = Settings()
