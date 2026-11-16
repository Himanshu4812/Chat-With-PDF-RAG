"""Database models and session management."""
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Database configuration
DATABASE_DIR = Path("data")
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/api.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


from typing import Optional, Any
from datetime import datetime

class PDFMetadata(Base):
    """PDF metadata table."""
    __tablename__ = "pdfs"

    pdf_id: Any = Column(String, primary_key=True)
    name: Any = Column(String, nullable=False)
    collection_name: Any = Column(String, unique=True, nullable=False)
    upload_timestamp: Any = Column(DateTime, nullable=False)
    doc_count: Any = Column(Integer, nullable=False)
    page_count: Any = Column(Integer, nullable=False)
    is_sample: Any = Column(Boolean, default=False)
    file_path: Any = Column(String)


class ChatSession(Base):
    """Chat session table."""
    __tablename__ = "chat_sessions"

    session_id: Any = Column(String, primary_key=True)
    created_at: Any = Column(DateTime, nullable=False)
    last_active: Any = Column(DateTime, nullable=False)


class ChatMessage(Base):
    """Chat message table."""
    __tablename__ = "messages"

    message_id: Any = Column(Integer, primary_key=True, autoincrement=True)
    session_id: Any = Column(String, nullable=False)
    role: Any = Column(String, nullable=False)
    content: Any = Column(String, nullable=False)
    sources: Any = Column(JSON)
    timestamp: Any = Column(DateTime, nullable=False)


# Create all tables
Base.metadata.create_all(bind=engine)
