from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


_HF_BASE = "https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R/resolve/main"
_QUESTIONS_URL = f"{_HF_BASE}/questions_QA_2024_v1.1.json"
_LEGAL_DOCS_URL = f"{_HF_BASE}/referred_legal_documents_QA_2024_v1.1.json"


_CACHE_DIR = Path(os.environ.get("HF_HOME", "/app/.cache/huggingface")) / "br_taxqa_r"


@dataclass
class Question:
    number: str
    summary: str
    text: str
    answer_cleaned: str
    references_explicit: List[str]
    references_implicit: List[str]
    all_ref_files: List[str]
    linked_questions: List[str]


@dataclass
class LegalChunk:
    doc_filename: str
    chunk_id: int
    text: str
    char_start: int
    char_end: int

def _download_if_needed(url: str, dest: Path) -> Path:

    if dest.exists():
        logger.info("Cache hit: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Baixando %s ...", url)

    try:
        import subprocess
        result = subprocess.run(
            ["curl", "-L", "-o", str(dest), url],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
    except (FileNotFoundError, RuntimeError):

        from urllib.request import urlretrieve
        urlretrieve(url, dest)
    logger.info("Download concluído: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def _normalize_to_filename(title: str) -> str:

    t = title.strip()
    if not t.endswith(".txt"):
        t = t + ".txt"
    return t


def _get_explicit_titles(formatted_refs) -> List[str]:
    if not formatted_refs:
        return []
    if isinstance(formatted_refs, list):
        return list({_normalize_to_filename(r.get("título", ""))
                     for r in formatted_refs if isinstance(r, dict) and r.get("título")})
    if isinstance(formatted_refs, dict):
        return list({_normalize_to_filename(t) for t in formatted_refs.get("título", []) if t})
    return []


def _get_implicit_titles(formatted_embedded_refs) -> List[str]:
    if not formatted_embedded_refs:
        return []
    if isinstance(formatted_embedded_refs, list):
        return list({_normalize_to_filename(r.get("título", ""))
                     for r in formatted_embedded_refs if isinstance(r, dict) and r.get("título")})
    if isinstance(formatted_embedded_refs, dict):
        return list({_normalize_to_filename(t) for t in formatted_embedded_refs.get("título", []) if t})
    return []


def _get_all_ref_files(all_formatted_refs) -> List[str]:
    if not all_formatted_refs:
        return []
    files = []
    if isinstance(all_formatted_refs, dict):
        for ref_list in all_formatted_refs.values():
            if isinstance(ref_list, list):
                for ref in ref_list:
                    if isinstance(ref, dict) and ref.get("file"):
                        files.append(ref["file"])
            elif isinstance(ref_list, dict) and ref_list.get("file"):
                files.append(ref_list["file"])
    return list(set(files))




def load_questions(
    url: str = _QUESTIONS_URL,
    cache_dir: Path = _CACHE_DIR,
) -> List[Question]:

    dest = cache_dir / "questions_QA_2024_v1.1.json"
    _download_if_needed(url, dest)

    logger.info("Parseando perguntas ...")
    with open(dest, encoding="utf-8") as f:
        raw_list = json.load(f)

    questions = []
    for row in raw_list:
        parts = row.get("answer_cleaned") or []
        answer_text = " ".join(parts).strip() if isinstance(parts, list) else str(parts)

        explicit = _get_explicit_titles(row.get("formatted_references"))
        implicit = _get_implicit_titles(row.get("formatted_embedded_references"))
        all_files = _get_all_ref_files(row.get("all_formatted_references"))

        questions.append(Question(
            number=str(row.get("question_number", "")),
            summary=str(row.get("question_summary", "")).strip().upper(),
            text=str(row.get("question_text", "")).strip(),
            answer_cleaned=answer_text,
            references_explicit=explicit,
            references_implicit=implicit,
            all_ref_files=all_files,
            linked_questions=list(row.get("linked_questions") or []),
        ))

    logger.info("Perguntas carregadas: %d", len(questions))
    return questions


def load_legal_docs(
    url: str = _LEGAL_DOCS_URL,
    cache_dir: Path = _CACHE_DIR,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Tuple[Dict[str, str], Dict[str, List[LegalChunk]]]:

    dest = cache_dir / "referred_legal_documents_QA_2024_v1.1.json"
    _download_if_needed(url, dest)

    logger.info("Parseando documentos legais ...")
    with open(dest, encoding="utf-8") as f:
        raw_list = json.load(f)

    doc_texts: Dict[str, str] = {}
    for row in raw_list:
        filename = row.get("filename") or row.get("file", "")
        text = row.get("filedata") or row.get("text", "")
        if filename and text:
            doc_texts[filename] = text

    logger.info("Documentos legais: %d", len(doc_texts))

    chunks_by_doc: Dict[str, List[LegalChunk]] = {}
    for filename, text in doc_texts.items():
        chunks_by_doc[filename] = _sliding_window_chunks(filename, text, chunk_size, chunk_overlap)

    total = sum(len(v) for v in chunks_by_doc.values())
    logger.info("Total de chunks: %d", total)
    return doc_texts, chunks_by_doc




def _sliding_window_chunks(
    filename: str,
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[LegalChunk]:
    chunks, step, i, chunk_id = [], chunk_size - overlap, 0, 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunk_text = text[i:end].strip()
        if chunk_text:
            chunks.append(LegalChunk(
                doc_filename=filename,
                chunk_id=chunk_id,
                text=chunk_text,
                char_start=i,
                char_end=end,
            ))
            chunk_id += 1
        if end == len(text):
            break
        i += step
    return chunks
