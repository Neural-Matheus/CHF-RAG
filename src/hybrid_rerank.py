from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_loader import LegalChunk, Question
from .baselines import ir_metrics, ir_metrics_at_ks

logger = logging.getLogger(__name__)

class BM25DocLevel:

    def __init__(self, all_chunks: List[LegalChunk]):
        from rank_bm25 import BM25Okapi
        self.all_chunks = all_chunks


        tokenized_corpus = [c.text.lower().split() for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 indexado: %d chunks", len(all_chunks))

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[LegalChunk, float]]:

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        doc_best = {}
        for i, c in enumerate(self.all_chunks):
            s = float(scores[i])
            if c.doc_filename not in doc_best or s > doc_best[c.doc_filename][1]:
                doc_best[c.doc_filename] = (i, s)
        ranked = sorted(doc_best.items(), key=lambda x: x[1][1], reverse=True)
        return [(self.all_chunks[ci], s) for _, (ci, s) in ranked[:k]]

    def retrieve_chunks(self, query: str, k: int = 100) -> List[Tuple[LegalChunk, float]]:
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        idx = np.argsort(scores)[::-1][:k]
        return [(self.all_chunks[i], float(scores[i])) for i in idx]




def reciprocal_rank_fusion(
    rankings: List[List[LegalChunk]],
    k: int = 60,
) -> List[Tuple[LegalChunk, float]]:
    chunk_scores = {}
    chunk_objs = {}
    for ranking in rankings:
        for rank_pos, chunk in enumerate(ranking, start=1):
            key = (chunk.doc_filename, chunk.chunk_id)
            chunk_objs[key] = chunk
            chunk_scores[key] = chunk_scores.get(key, 0.0) + 1.0 / (k + rank_pos)

    sorted_keys = sorted(chunk_scores.keys(), key=lambda k_: -chunk_scores[k_])
    return [(chunk_objs[key], chunk_scores[key]) for key in sorted_keys]




class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Carregando reranker '%s' em %s ...", model_name, device)
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.model_name = model_name
        self.batch_size = batch_size
        self.n_calls = 0
        self.total_pairs = 0
        self.total_time = 0.0

    def rerank(
        self,
        query: str,
        candidates: List[LegalChunk],
        top_k: int = 10,
    ) -> List[Tuple[LegalChunk, float]]:
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        t0 = time.time()
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
        )
        elapsed = time.time() - t0
        self.n_calls += 1
        self.total_pairs += len(pairs)
        self.total_time += elapsed

        order = np.argsort(scores)[::-1][:top_k]
        return [(candidates[i], float(scores[i])) for i in order]

    def stats(self) -> Dict[str, float]:
        return {
            "n_calls": self.n_calls,
            "total_pairs_scored": self.total_pairs,
            "total_time_s": round(self.total_time, 2),
            "avg_pairs_per_call": (self.total_pairs / max(self.n_calls, 1)),
            "avg_time_per_pair_ms": (
                1000 * self.total_time / max(self.total_pairs, 1)
            ),
        }




def evaluate_bm25_pure(
    bm25: BM25DocLevel,
    questions: List[Question],
    embedder_label: str,
    k: int = 10,
    ks: List[int] = None,
) -> List[Dict]:
    rows = []
    k_max = max(ks) if ks else k
    for q in questions:
        if not q.all_ref_files:
            continue
        retrieved = [c for c, _ in bm25.retrieve(q.text, k=k_max)]
        row = {
            "question_number": q.number,
            "strategy": "bm25",
            "embedder": embedder_label,
            **ir_metrics(retrieved[:k], q),
        }
        if ks:
            row.update(ir_metrics_at_ks(retrieved, q, ks=ks))
        rows.append(row)
    return rows


def evaluate_with_rerank(
    candidate_fn,
    reranker: CrossEncoderReranker,
    questions: List[Question],
    strategy_name: str,
    embedder_label: str,
    n_candidates: int = 100,
    k: int = 10,
    ks: List[int] = None,
) -> List[Dict]:
    rows = []
    k_max = max(ks) if ks else k
    for q in questions:
        if not q.all_ref_files:
            continue
        candidates = candidate_fn(q, n_candidates)
        if not candidates:
            row = {
                "question_number": q.number,
                "strategy": strategy_name,
                "embedder": embedder_label,
                "f1_k": 0.0, "precision_k": 0.0, "recall_k": 0.0,
            }
            if ks:
                for kk in ks:
                    row.update({
                        f"f1_{kk}": 0.0,
                        f"precision_{kk}": 0.0,
                        f"recall_{kk}": 0.0,
                    })
            rows.append(row)
            continue
        reranked = reranker.rerank(q.text, candidates, top_k=k_max)
        retrieved = [c for c, _ in reranked]
        row = {
            "question_number": q.number,
            "strategy": strategy_name,
            "embedder": embedder_label,
            "n_candidates": len(candidates),
            **ir_metrics(retrieved[:k], q),
        }
        if ks:
            row.update(ir_metrics_at_ks(retrieved, q, ks=ks))
        rows.append(row)
    return rows
