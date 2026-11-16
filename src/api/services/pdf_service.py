"""PDF processing service."""
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session

from langchain_ollama import OllamaEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from ...core.document import DocumentProcessor
    from ...core.embeddings import VectorStore
    from ..database import PDFMetadata
    from ..config import settings
except (ImportError, ValueError):
    from src.core.document import DocumentProcessor
    from src.core.embeddings import VectorStore
    from src.api.database import PDFMetadata
    from src.api.config import settings


class PDFService:
    """Service for PDF operations."""

    def __init__(self):
        """Initialize PDF service."""
        self.doc_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.vector_store = VectorStore(
            embedding_model=settings.EMBEDDING_MODEL,
            persist_directory=settings.VECTOR_DB_DIR
        )
        self.storage_dir = Path(settings.PDF_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def upload_and_process(
        self,
        file: UploadFile,
        db: Session
    ) -> PDFMetadata:
        """Upload and process a PDF file.

        Args:
            file: Uploaded PDF file
            db: Database session

        Returns:
            PDFMetadata: Metadata for the processed PDF
        """
        # Extract safe filename (FastAPI UploadFile.filename can be Optional[str])
        filename = file.filename or "document.pdf"

        # Generate unique ID
        pdf_id = self._generate_pdf_id(filename)

        # Save file
        file_path = self.storage_dir / f"{pdf_id}_{filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process PDF (with OCR fallback for scanned docs when enabled)
        if settings.ENABLE_OCR_FALLBACK:
            documents = self.doc_processor.extract_with_ocr_fallback(
                file_path, min_chars=settings.OCR_MIN_CHARS
            )
        else:
            documents = self.doc_processor.load_pdf(file_path)
        chunks = self.doc_processor.split_documents(documents)

        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "pdf_id": pdf_id,
                "pdf_name": filename,
                "chunk_index": i,
                "source_file": filename
            })

        # Create vector DB collection
        collection_name = f"pdf_{abs(hash(filename + pdf_id))}"
        vector_db = self.vector_store.create_vector_db(
            documents=chunks,
            collection_name=collection_name
        )

        # Store metadata in database
        pdf_metadata = PDFMetadata(
            pdf_id=pdf_id,
            name=filename,
            collection_name=collection_name,
            upload_timestamp=datetime.now(),
            doc_count=len(chunks),
            page_count=len(documents),
            is_sample=False,
            file_path=str(file_path)
        )
        db.add(pdf_metadata)
        db.commit()
        db.refresh(pdf_metadata)

        return pdf_metadata

    def list_pdfs(self, db: Session) -> List[PDFMetadata]:
        """List all PDFs.

        Args:
            db: Database session

        Returns:
            List of PDF metadata
        """
        return db.query(PDFMetadata).all()

    def get_pdf(self, pdf_id: str, db: Session) -> Optional[PDFMetadata]:
        """Get single PDF metadata.

        Args:
            pdf_id: PDF identifier
            db: Database session

        Returns:
            PDF metadata or None
        """
        return db.query(PDFMetadata).filter(PDFMetadata.pdf_id == pdf_id).first()

    def delete_pdf(self, pdf_id: str, db: Session) -> bool:
        """Delete PDF and its collection.

        Args:
            pdf_id: PDF identifier
            db: Database session

        Returns:
            True if deleted successfully, False otherwise
        """
        pdf = self.get_pdf(pdf_id, db)
        if not pdf:
            return False

        # Delete vector collection
        collection_name = getattr(pdf, "collection_name", None) or str(pdf.collection_name)
        embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        vector_db = Chroma(
            persist_directory=settings.VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        vector_db.delete_collection()

        # Delete file if it exists
        file_path = getattr(pdf, "file_path", None)
        if file_path and os.path.exists(str(file_path)):
            os.remove(str(file_path))

        # Delete metadata from database
        db.delete(pdf)
        db.commit()

        return True

    def _generate_pdf_id(self, filename: Optional[str] = None) -> str:
        """Generate unique PDF ID.

        Args:
            filename: Original filename

        Returns:
            Unique PDF identifier
        """
        safe_name = filename or "document.pdf"
        timestamp = datetime.now().isoformat()
        return f"pdf_{abs(hash(safe_name + timestamp))}"
