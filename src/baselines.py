from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from .data_loader import LegalChunk, Question

logger = logging.getLogger(__name__)

def ir_metrics(
    retrieved: List[LegalChunk],
    question: Question,
) -> Dict[str, float]:
    gt = set(question.all_ref_files)
    if not gt:
        return {"precision_k": 0.0, "recall_k": 0.0, "f1_k": 0.0}

    retrieved_files = set(c.doc_filename for c in retrieved)
    tp = len(retrieved_files & gt)
    precision = tp / len(retrieved_files) if retrieved_files else 0.0
    recall    = tp / len(gt)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"precision_k": precision, "recall_k": recall, "f1_k": f1}


def ir_metrics_at_ks(retrieved, question, ks=[5, 10, 20]):
    out = {}
    for k in ks:
        m = ir_metrics(retrieved[:k], question)
        out[f"precision_at_{k}"] = m["precision_k"]
        out[f"recall_at_{k}"] = m["recall_k"]
        out[f"f1_at_{k}"] = m["f1_k"]
    return out




def evaluate_cosine(
    questions: List[Question],
    question_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    all_chunks: List[LegalChunk],
    embedder_label: str,
    k: int = 10,
) -> List[Dict]:
    rows = []
    for i, question in enumerate(questions):
        if not question.all_ref_files:
            continue
        scores = chunk_embeddings @ question_embeddings[i]
        top_idx = np.argsort(scores)[::-1][:k]
        retrieved = [all_chunks[j] for j in top_idx]
        m = ir_metrics(retrieved, question)
        rows.append({
            "question_number": question.number,
            "strategy": "cosine",
            "embedder": embedder_label,
            "reference": "Lewis et al. NeurIPS 2020",
            "faithfulness": None,
            "xai_explanation": None,
            **m,
        })
    logger.info("CosSim [%s]: %d questions", embedder_label, len(rows))
    return rows


def evaluate_cosine_doc_level(
    questions: List[Question],
    question_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    all_chunks: List[LegalChunk],
    embedder_label: str,
    k: int = 10,
    ks: List[int] = None,
) -> List[Dict]:
    rows = []
    k_max = max(ks) if ks else k
    for i, question in enumerate(questions):
        if not question.all_ref_files:
            continue
        scores = chunk_embeddings @ question_embeddings[i]


        doc_best_chunk: Dict[str, Tuple[int, float]] = {}
        for j, c in enumerate(all_chunks):
            sc = float(scores[j])
            cur = doc_best_chunk.get(c.doc_filename)
            if cur is None or sc > cur[1]:
                doc_best_chunk[c.doc_filename] = (j, sc)


        ranked = sorted(doc_best_chunk.items(), key=lambda x: -x[1][1])[:k_max]
        retrieved = [all_chunks[idx] for _, (idx, _) in ranked]
        row = {
            "question_number": question.number,
            "strategy": "cosine_doc_level",
            "embedder": embedder_label,
            "reference": "Lewis et al. NeurIPS 2020 (doc-aggregated)",
            "faithfulness": None,
            "xai_explanation": None,
            **ir_metrics(retrieved[:k], question),
        }
        if ks:
            row.update(ir_metrics_at_ks(retrieved, question, ks=ks))
        rows.append(row)
    logger.info("CosSim doc-level [%s]: %d questions", embedder_label, len(rows))
    return rows
