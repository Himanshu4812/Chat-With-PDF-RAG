"""Benchmark RAG query performance against the running FastAPI backend.

Usage:
    python scripts/benchmark.py --out benchmark_results/baseline_phi4.json \
        --model phi4-mini:3.8b --pdf-id pdf_8647084836463591749

Records real measurements only. Supports the non-streaming /query endpoint
(baseline and Stage 1) and, once implemented, the /query/stream endpoint.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.api.config import settings

    BACKEND_CONFIG = {
        "retrieval_k": settings.RETRIEVAL_K,
        "retrieval_max_chunks": settings.RETRIEVAL_MAX_CHUNKS,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "multi_query": settings.USE_MULTI_QUERY,
        "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
        "num_ctx": settings.NUM_CTX,
        "num_predict": settings.NUM_PREDICT,
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        "temperature": settings.LLM_TEMPERATURE,
    }
except Exception:  # pragma: no cover - benchmark still works without import
    BACKEND_CONFIG = {}

API_BASE = "http://127.0.0.1:8001"
OLLAMA_BASE = "http://127.0.0.1:11434"

QUESTIONS = [
    {
        "id": "q1_title",
        "category": "simple_factual",
        "question": "What is the title of the project described in this document?",
    },
    {
        "id": "q2_objectives",
        "category": "multi_section",
        "question": (
            "What are the objectives of the project, and which technologies are "
            "used for the frontend and backend?"
        ),
    },
    {
        "id": "q3_numbers",
        "category": "specific_numbers",
        "question": (
            "How many user stories should be prepared, and in what format?"
        ),
    },
    {
        "id": "q4_not_in_pdf",
        "category": "not_in_pdf",
        "question": (
            "What does the document say about the migration patterns of "
            "monarch butterflies?"
        ),
    },
    {
        "id": "q5_summarize",
        "category": "summarize",
        "question": "Summarize this project in 3-4 sentences.",
    },
]

REFUSAL_PATTERNS = re.compile(
    r"(do not contain|does not contain|not contain|no information|"
    r"cannot find|could not find|not mentioned|not provided|"
    r"do not have|does not mention|not available in)",
    re.IGNORECASE,
)


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_snapshot() -> dict:
    """Return physical memory stats (Windows; graceful fallback)."""
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {
            "total_gb": round(stat.ullTotalPhys / 1024**3, 2),
            "available_gb": round(stat.ullAvailPhys / 1024**3, 2),
            "used_gb": round(
                (stat.ullTotalPhys - stat.ullAvailPhys) / 1024**3, 2
            ),
            "load_percent": stat.dwMemoryLoad,
        }
    except Exception as exc:  # pragma: no cover - non-Windows fallback
        return {"error": str(exc)}


def http_json(url: str, payload: dict | None = None, timeout: int = 600):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_ps() -> str:
    try:
        out = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def warmup(model: str) -> dict:
    """Load chat + embedding models so Q1 measures steady-state latency.

    Uses the same num_ctx as real queries; a different context size would
    force Ollama to reload the model inside the first measured question.
    """
    result = {}
    num_ctx = BACKEND_CONFIG.get("num_ctx", 2048)
    keep_alive = BACKEND_CONFIG.get("keep_alive", "30m")
    t0 = time.perf_counter()
    try:
        http_json(
            f"{OLLAMA_BASE}/api/generate",
            {
                "model": model,
                "prompt": "OK",
                "stream": False,
                "options": {"num_predict": 1, "num_ctx": num_ctx},
                "keep_alive": keep_alive,
            },
            timeout=300,
        )
        result["chat_warmup_s"] = round(time.perf_counter() - t0, 2)
    except Exception as exc:
        result["chat_warmup_error"] = str(exc)

    t0 = time.perf_counter()
    try:
        http_json(
            f"{OLLAMA_BASE}/api/embed",
            {
                "model": "nomic-embed-text",
                "input": "warmup",
                "keep_alive": keep_alive,
            },
            timeout=300,
        )
        result["embed_warmup_s"] = round(time.perf_counter() - t0, 2)
    except Exception as exc:
        result["embed_warmup_error"] = str(exc)
    return result


def run_question(model: str, pdf_ids: list[str], item: dict) -> dict:
    payload = {"question": item["question"], "model": model}
    if pdf_ids:
        payload["pdf_ids"] = pdf_ids

    record = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
    }
    t0 = time.perf_counter()
    try:
        resp = http_json(f"{API_BASE}/api/v1/query", payload, timeout=600)
        elapsed = time.perf_counter() - t0
        answer = resp.get("answer") or ""
        metadata = resp.get("metadata") or {}
        record.update(
            {
                "total_seconds": round(elapsed, 2),
                "ttft_seconds": round(elapsed, 2),  # non-streaming: no TTFT
                "answer_length": len(answer),
                "sources_count": len(resp.get("sources") or []),
                "chunks_retrieved": metadata.get("chunks_retrieved"),
                "reasoning_steps": len(metadata.get("reasoning_steps") or []),
                "timings": metadata.get("timings"),
                "refusal_detected": bool(REFUSAL_PATTERNS.search(answer)),
                "answer": answer,
                "answer_preview": answer[:220].replace("\n", " "),
                "error": None,
            }
        )
        if item["category"] == "not_in_pdf":
            record["refusal_correct"] = record["refusal_detected"]
        else:
            record["refusal_correct"] = None
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - t0
        body = exc.read().decode("utf-8", errors="replace")
        record.update(
            {
                "total_seconds": round(elapsed, 2),
                "error": f"HTTP {exc.code}: {body[:300]}",
            }
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        record.update(
            {"total_seconds": round(elapsed, 2), "error": str(exc)}
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RAG backend")
    parser.add_argument(
        "--out",
        default="benchmark_results/run.json",
        help="Path to output JSON",
    )
    parser.add_argument("--model", default="phi4-mini:3.8b")
    parser.add_argument(
        "--pdf-id",
        action="append",
        default=[],
        help="PDF id to query (repeatable). Omit to use all PDFs.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip model warm-up before measuring",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="Label stored in the output JSON",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional question ids to run (e.g. --only q1_title q4_not_in_pdf)",
    )
    args = parser.parse_args()

    questions = QUESTIONS
    if args.only:
        known = {q["id"] for q in QUESTIONS}
        unknown = [q for q in args.only if q not in known]
        if unknown:
            print(f"FATAL: unknown question ids: {unknown}")
            return 2
        questions = [q for q in QUESTIONS if q["id"] in args.only]

    try:
        health = http_json(f"{API_BASE}/api/v1/health", timeout=10)
    except Exception as exc:
        print(f"FATAL: backend not reachable at {API_BASE}: {exc}")
        return 2
    if not health.get("ollama_connected"):
        print("FATAL: Ollama not connected according to backend health")
        return 2

    pdfs = http_json(f"{API_BASE}/api/v1/pdfs", timeout=30)
    pdf_lookup = {p["pdf_id"]: p for p in pdfs}
    selected = args.pdf_id or []
    for pid in selected:
        if pid not in pdf_lookup:
            print(f"FATAL: PDF id not found: {pid}")
            return 2

    print(f"Backend OK. Model: {args.model}")
    for pid in selected:
        p = pdf_lookup[pid]
        print(
            f"PDF: {p['name']} | chunks={p['doc_count']} pages={p['page_count']}"
        )
    print()

    mem_before = memory_snapshot()
    ps_before = ollama_ps()
    warm = {"skipped": True} if args.no_warmup else warmup(args.model)
    if not args.no_warmup:
        print(f"Warm-up: {warm}")
        print()

    records = []
    for item in questions:
        print(f"[{item['id']}] {item['question'][:70]}...", flush=True)
        rec = run_question(args.model, selected, item)
        records.append(rec)
        if rec.get("error"):
            print(f"  ERROR: {rec['error']}")
        else:
            print(
                f"  {rec['total_seconds']}s | chars={rec['answer_length']} "
                f"| chunks={rec['chunks_retrieved']} "
                f"| sources={rec['sources_count']} "
                f"| refusal={rec['refusal_detected']}"
            )

    mem_after = memory_snapshot()
    ps_after = ollama_ps()

    ok = [r for r in records if not r.get("error")]
    totals = [r["total_seconds"] for r in ok]
    refusal_rec = next(
        (r for r in records if r["id"] == "q4_not_in_pdf"), None
    )
    summary = {
        "questions_run": len(records),
        "questions_ok": len(ok),
        "total_seconds_sum": round(sum(totals), 2) if totals else None,
        "mean_seconds": round(statistics.mean(totals), 2) if totals else None,
        "median_seconds": (
            round(statistics.median(totals), 2) if totals else None
        ),
        "min_seconds": round(min(totals), 2) if totals else None,
        "max_seconds": round(max(totals), 2) if totals else None,
        "refusal_correct": (
            refusal_rec.get("refusal_correct") if refusal_rec else None
        ),
    }

    output = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "non-streaming /api/v1/query",
        "model": args.model,
        "provider_mode": "ollama",
        "pdf_ids": selected,
        "pdfs": [
            {
                "pdf_id": pid,
                "name": pdf_lookup[pid]["name"],
                "chunks": pdf_lookup[pid]["doc_count"],
                "pages": pdf_lookup[pid]["page_count"],
            }
            for pid in selected
        ],
        "config": BACKEND_CONFIG,
        "warmup": warm,
        "memory_before": mem_before,
        "memory_after": mem_after,
        "ollama_ps_before": ps_before,
        "ollama_ps_after": ps_after,
        "summary": summary,
        "questions": records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 78)
    print(
        f"{'ID':<15}{'CATEGORY':<18}{'SECONDS':>9}{'CHARS':>8}"
        f"{'CHUNKS':>8}{'REFUSAL':>9}"
    )
    print("-" * 78)
    for r in records:
        if r.get("error"):
            print(f"{r['id']:<15}{r['category']:<18}  ERROR: {r['error'][:40]}")
            continue
        print(
            f"{r['id']:<15}{r['category']:<18}{r['total_seconds']:>9.1f}"
            f"{r['answer_length']:>8}{str(r['chunks_retrieved']):>8}"
            f"{str(r['refusal_detected']):>9}"
        )
    print("-" * 78)
    print(
        f"TOTAL {summary['total_seconds_sum']}s | MEAN {summary['mean_seconds']}s "
        f"| MEDIAN {summary['median_seconds']}s | refused_correctly="
        f"{summary['refusal_correct']}"
    )
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
