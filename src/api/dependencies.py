"""FastAPI dependencies for dependency injection."""
from functools import lru_cache

from sqlalchemy.orm import Session
from .database import SessionLocal
from .services.pdf_service import PDFService
from .services.rag_service import RAGService


def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache(maxsize=1)
def get_pdf_service() -> PDFService:
    """PDF service dependency (singleton)."""
    return PDFService()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """RAG service dependency (singleton)."""
    return RAGService()
