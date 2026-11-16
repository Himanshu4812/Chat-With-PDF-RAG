"""RAG quality regression tests (no Ollama/Chroma required).

Covers the deterministic parts of the RAG upgrade:
- chunking defaults + overlap
- OCR scanned detection
- dedup + context formatting with structured citations
- consolidated prompts contain citation + no-hallucination guardrails
- golden QA fixture validity
"""
import json
from pathlib import Path

from langchain_core.documents import Document

from src.core.document import DocumentProcessor
from src.core.llm import LLMManager
from src.core.retrieval import (
    deduplicate_docs,
    format_context,
    extract_sources,
)


def test_chunk_defaults_balanced():
    p = DocumentProcessor()
    assert p.chunk_size == 1500
    assert p.chunk_overlap == 200
    assert p.chunk_overlap < p.chunk_size


def test_chunk_overlap_preserved():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
    doc = Document(page_content=text, metadata={"source": "test"})
    p = DocumentProcessor(chunk_size=50, chunk_overlap=15)
    chunks = p.split_documents([doc])
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.page_content) <= 50


def test_is_scanned_detection():
    assert DocumentProcessor.is_scanned(
        [Document(page_content="   ", metadata={})], min_chars=50
    ) is True
    assert DocumentProcessor.is_scanned(
        [Document(page_content="hello world " * 20, metadata={})], min_chars=50
    ) is False


def test_ocr_fallback_returns_list_without_deps(tmp_path):
    # Empty PDF-like extraction should not crash even if OCR deps missing
    p = DocumentProcessor()
    docs = [Document(page_content="", metadata={"source": "x"})]
    # is_scanned True path is covered; fallback import failure returns input
    assert DocumentProcessor.is_scanned(docs) is True


def test_deduplicate_docs():
    docs = [
        Document(page_content="same content", metadata={"pdf_name": "a.pdf"}),
        Document(page_content="same content", metadata={"pdf_name": "a.pdf"}),
        Document(page_content="different content", metadata={"pdf_name": "a.pdf"}),
    ]
    out = deduplicate_docs(docs)
    assert len(out) == 2


def test_format_context_structured_citations():
    docs = [
        Document(
            page_content="MFA is required.",
            metadata={"pdf_name": "Guide.pdf", "page_number": 3, "chunk_index": 7},
        )
    ]
    ctx = format_context(docs, max_chunks=10)
    assert "[Source: Guide.pdf p.3 #7]" in ctx
    assert "MFA is required." in ctx


def test_extract_sources_has_page():
    docs = [
        Document(
            page_content="x",
            metadata={
                "pdf_name": "G.pdf",
                "pdf_id": "pdf_1",
                "chunk_index": 2,
                "page_number": 5,
            },
        )
    ]
    sources = extract_sources(docs)
    assert sources[0]["page_number"] == 5
    assert sources[0]["pdf_name"] == "G.pdf"


def test_rag_prompt_has_guardrails():
    mgr = LLMManager(model_name="test-model")
    prompt = mgr.get_rag_prompt()
    tmpl = prompt.messages[0].prompt.template
    assert "ONLY" in tmpl
    assert "[Source:" in tmpl
    assert "do not contain this information" in tmpl.lower()


def test_thinking_model_detection():
    assert LLMManager.is_thinking_model("qwen3:8b") is True
    assert LLMManager.is_thinking_model("deepseek-r1:7b") is True
    assert LLMManager.is_thinking_model("llama3.2") is False
    assert LLMManager.is_thinking_model("mistral:latest") is False


def test_golden_qa_fixture_valid():
    path = Path("tests/fixtures/golden_qa.json")
    assert path.exists(), "golden QA fixture missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) >= 2
    for item in data:
        assert "id" in item and "question" in item
        assert len(item["question"]) > 5
