"""Document processing functionality."""
import logging
from pathlib import Path
from typing import List
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF document loading and processing."""

    # Balanced defaults: 1500 chars (~300-400 tokens) for precision,
    # 200 overlap to preserve sentence continuity.
    # Legacy value was 7500/100 which hurt retrieval precision.
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF document using pdfplumber and return LangChain Document objects."""
        try:
            logger.info(f"Loading PDF from {file_path}")
            documents = []
            with pdfplumber.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append(Document(
                            page_content=text.strip(),
                            metadata={
                                "source": str(file_path),
                                "page_number": i + 1,
                                "total_pages": len(pdf.pages)
                            }
                        ))
            logger.info(f"Extracted {len(documents)} pages from PDF")
            # OCR fallback detection: flag scanned docs with almost no text
            total_chars = sum(len(d.page_content) for d in documents)
            if total_chars == 0:
                logger.warning(
                    f"No extractable text in {file_path} "
                    "(likely scanned/image-only). Consider OCR fallback "
                    "via extract_with_ocr_fallback()."
                )
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    @staticmethod
    def is_scanned(documents: List[Document], min_chars: int = 50) -> bool:
        """Return True if extracted text looks like a scanned PDF."""
        total = sum(len(d.page_content.strip()) for d in documents)
        return total < min_chars

    def extract_with_ocr_fallback(
        self, file_path: Path, min_chars: int = 50
    ) -> List[Document]:
        """Load PDF with OCR fallback for scanned/image-only pages.

        Tries pdfplumber first; if total chars < min_chars, attempts
        pdf2image + pytesseract page-by-page. Both are optional deps —
        if unavailable, returns whatever pdfplumber extracted with a
        warning so ingestion never hard-crashes.
        """
        documents = self.load_pdf(file_path)
        total_chars = sum(len(d.page_content) for d in documents)
        if total_chars >= min_chars:
            return documents

        logger.warning(
            f"Scanned document detected ({total_chars} chars). "
            "Attempting OCR fallback."
        )
        try:
            from pdf2image import convert_from_path  # type: ignore
            import pytesseract  # type: ignore
        except ImportError:
            logger.warning(
                "OCR deps missing (pdf2image/pytesseract). "
                "Install with: pip install pdf2image pytesseract. "
                "Returning raw extraction."
            )
            return documents

        try:
            images = convert_from_path(str(file_path))
            ocr_docs: List[Document] = []
            # Re-read total_pages from first pass if available
            total_pages = len(images)
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img) or ""
                text = text.strip()
                if text:
                    ocr_docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file_path),
                                "page_number": i + 1,
                                "total_pages": total_pages,
                                "ocr": True,
                            },
                        )
                    )
            if ocr_docs:
                logger.info(f"OCR extracted {len(ocr_docs)} pages")
                return ocr_docs
            logger.warning("OCR produced no text; returning raw extraction.")
            return documents
        except Exception as e:
            logger.error(f"OCR fallback failed: {e}. Returning raw extraction.")
            return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            logger.info("Splitting documents into chunks")
            return self.splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise 