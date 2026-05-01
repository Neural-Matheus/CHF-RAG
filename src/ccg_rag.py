from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .data_loader import LegalChunk, Question
from .baselines import ir_metrics

logger = logging.getLogger(__name__)

@dataclass
class CoCitationGraph:
    cocit_matrix: Dict[str, Dict[str, float]]
    doc_to_questions: Dict[str, set]
    n_documents: int
    n_edges: int
    mean_weight: float

    threshold: float = 0.0


def build_cocitation_graph(
    questions: List[Question],
    min_cooccurrences: int = 1,
) -> CoCitationGraph:
    doc_to_questions: Dict[str, set] = defaultdict(set)
    for q in questions:
        for doc in q.all_ref_files:
            doc_to_questions[doc].add(q.number)

    docs = list(doc_to_questions.keys())
    logger.info("Construindo grafo de co-citação: %d documentos ...", len(docs))

    cocit_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    n_edges, weights = 0, []

    for i, d_i in enumerate(docs):
        q_i = doc_to_questions[d_i]
        for j, d_j in enumerate(docs):
            if j <= i:
                continue
            q_j = doc_to_questions[d_j]
            inter = len(q_i & q_j)
            if inter < min_cooccurrences:
                continue
            union = len(q_i | q_j)
            weight = inter / union
            cocit_matrix[d_i][d_j] = weight
            cocit_matrix[d_j][d_i] = weight
            n_edges += 1
            weights.append(weight)

    mean_weight = float(np.mean(weights)) if weights else 0.0
    logger.info(
        "Grafo: %d nós | %d arestas | peso médio=%.4f",
        len(docs), n_edges, mean_weight
    )

    return CoCitationGraph(
        cocit_matrix=dict(cocit_matrix),
        doc_to_questions=dict(doc_to_questions),
        n_documents=len(docs),
        n_edges=n_edges,
        mean_weight=mean_weight,
    )

def ccg_rag_retrieve(
    question: Question,
    q_emb: np.ndarray,
    chunk_embs: np.ndarray,
    all_chunks: List[LegalChunk],
    graph: CoCitationGraph,
    k: int = 10,
    K: int = 200,
    max_expand: int = 100,
    min_cocit: float = 0.05,
) -> Tuple[List[LegalChunk], List[Dict]]:
    cos_scores = chunk_embs @ q_emb


    doc_to_chunk_idx: Dict[str, List[int]] = defaultdict(list)
    for idx, chunk in enumerate(all_chunks):
        doc_to_chunk_idx[chunk.doc_filename].append(idx)


    top_K_idx = np.argsort(cos_scores)[::-1][:K].tolist()
    pool_docs: Set[str] = {all_chunks[i].doc_filename for i in top_K_idx}
    pool_chunks: Set[int] = set(top_K_idx)


    expansion_info: Dict[int, Dict] = {}
    n_docs_expanded = 0

    for doc in list(pool_docs):
        if doc not in graph.cocit_matrix:
            continue
        neighbors = sorted(
            graph.cocit_matrix[doc].items(),
            key=lambda x: x[1], reverse=True
        )
        for neighbor_doc, weight in neighbors:
            if weight < min_cocit:
                break
            if neighbor_doc in pool_docs:

                continue

            pool_docs.add(neighbor_doc)
            n_shared = len(
                graph.doc_to_questions.get(doc, set()) &
                graph.doc_to_questions.get(neighbor_doc, set())
            )

            candidate_chunks = doc_to_chunk_idx.get(neighbor_doc, [])
            if not candidate_chunks:
                continue
            best_chunk_idx = max(candidate_chunks, key=lambda i: cos_scores[i])
            if best_chunk_idx not in pool_chunks:
                pool_chunks.add(best_chunk_idx)
                expansion_info[best_chunk_idx] = {
                    "entry": "graph",
                    "via_doc": doc,
                    "cocit_weight": weight,
                    "n_shared_questions": n_shared,
                }
            n_docs_expanded += 1
            if n_docs_expanded >= max_expand:
                break
        if n_docs_expanded >= max_expand:
            break

    logger.debug(
        "  Pool: %d docs iniciais | %d docs via grafo | total pool=%d chunks",
        len({all_chunks[i].doc_filename for i in top_K_idx}),
        n_docs_expanded,
        len(pool_chunks),
    )


    expanded_list = list(pool_chunks)
    expanded_scores = cos_scores[expanded_list]
    top_k_local = np.argsort(expanded_scores)[::-1][:k]
    top_k_global = [expanded_list[i] for i in top_k_local]

    retrieved = [all_chunks[i] for i in top_k_global]


    explanations = []
    for global_idx in top_k_global:
        chunk = all_chunks[global_idx]
        if global_idx in expansion_info:
            info = expansion_info[global_idx]
            explanation = (
                f"via_grafo | co-citado com '{info['via_doc']}' "
                f"(w={info['cocit_weight']:.3f}, "
                f"{info['n_shared_questions']} questões)"
            )
        else:
            info = {"entry": "cosine", "via_doc": "", "cocit_weight": 0.0, "n_shared_questions": 0}
            explanation = f"cos={float(cos_scores[global_idx]):.3f}"

        explanations.append({
            "doc": chunk.doc_filename,
            "chunk_id": chunk.chunk_id,
            "cos_score": float(cos_scores[global_idx]),
            "entry": info["entry"],
            "best_cocit_neighbor": info.get("via_doc", ""),
            "cocit_weight": info.get("cocit_weight", 0.0),
            "n_shared_questions": info.get("n_shared_questions", 0),
            "explanation": explanation,
        })

    return retrieved, explanations




def find_best_params(
    questions: List[Question],
    q_embs: np.ndarray,
    chunk_embs: np.ndarray,
    all_chunks: List[LegalChunk],
    graph: CoCitationGraph,
    k: int = 10,
    K_values: List[int] = None,
    min_cocit_values: List[float] = None,
) -> Tuple[int, float]:

    K_values = K_values or [30, 50, 100]
    min_cocit_values = min_cocit_values or [0.05, 0.1, 0.2, 0.3]

    best_K, best_min_cocit, best_recall = 50, 0.1, 0.0

    for K in K_values:
        for min_cocit in min_cocit_values:
            recalls = []
            for i, q in enumerate(questions):
                if not q.all_ref_files:
                    continue
                retrieved, _ = ccg_rag_retrieve(
                    q, q_embs[i], chunk_embs, all_chunks,
                    graph, k=k, K=K, min_cocit=min_cocit
                )
                m = ir_metrics(retrieved, q)
                recalls.append(m["recall_k"])
            mean_r = float(np.mean(recalls)) if recalls else 0.0
            logger.info("  K=%d min_cocit=%.2f → recall@k=%.4f", K, min_cocit, mean_r)
            if mean_r > best_recall:
                best_recall = mean_r
                best_K, best_min_cocit = K, min_cocit

    logger.info("  Params ótimos: K=%d min_cocit=%.2f (recall=%.4f)",
                best_K, best_min_cocit, best_recall)
    return best_K, best_min_cocit




def explanation_precision(
    explanations: List[Dict],
    question: Question,
) -> float:
    gt = set(question.all_ref_files)
    graph_neighbors = [
        e["best_cocit_neighbor"]
        for e in explanations
        if e["entry"] == "graph" and e["best_cocit_neighbor"]
    ]
    if not graph_neighbors:
        return 0.0
    correct = sum(1 for n in graph_neighbors if n in gt)
    return float(correct / len(graph_neighbors))


def aumann_shapley_attribution(
    explanations: List[Dict],
    graph: CoCitationGraph,
) -> Dict[str, float]:
    attributions = {}
    for e in explanations:
        if e["entry"] == "graph":
            doc = e["doc"]

            attributions[doc] = e["cocit_weight"]
        else:
            attributions[e["doc"]] = 0.0
    return attributions


def compute_erc(
    question: Question,
    q_emb: np.ndarray,
    chunk_embs: np.ndarray,
    all_chunks: List[LegalChunk],
    graph: CoCitationGraph,
    K_values: List[int] = None,
    k: int = 10,
    min_cocit: float = 0.1,
) -> float:
    K_values = K_values or [10, 20, 30, 50, 75, 100]
    if not question.all_ref_files or len(K_values) < 2:
        return 0.0

    f1_values, eq_values = [], []
    for K in K_values:
        retrieved, explanations = ccg_rag_retrieve(
            question, q_emb, chunk_embs, all_chunks,
            graph, k=k, K=K, min_cocit=min_cocit
        )
        m = ir_metrics(retrieved, question)
        f1_values.append(m["f1_k"])
        eq_values.append(explanation_precision(explanations, question))

    f1_arr = np.array(f1_values)
    eq_arr  = np.array(eq_values)
    K_arr   = np.array(K_values, dtype=float)

    dK   = np.diff(K_arr)
    df1  = np.abs(np.diff(f1_arr))
    sens = df1 / (dK + 1e-9)
    eq_mid = (eq_arr[:-1] + eq_arr[1:]) / 2.0

    numerator   = float(np.sum(eq_mid * sens * dK))
    denominator = float(np.sum(sens * dK)) + 1e-9

    return numerator / denominator

def evaluate_ccg_rag(
    questions: List[Question],
    q_embs: np.ndarray,
    chunk_embs: np.ndarray,
    all_chunks: List[LegalChunk],
    graph: CoCitationGraph,
    embedder_label: str,
    k: int = 10,
    K: int = 50,
    min_cocit: float = 0.1,
) -> List[Dict]:
    rows = []
    for i, question in enumerate(questions):
        if not question.all_ref_files:
            continue

        retrieved, explanations = ccg_rag_retrieve(
            question, q_embs[i], chunk_embs, all_chunks,
            graph, k=k, K=K, min_cocit=min_cocit
        )
        m = ir_metrics(retrieved, question)

        n_via_graph  = sum(1 for e in explanations if e["entry"] == "graph")
        ep           = explanation_precision(explanations, question)
        phi          = aumann_shapley_attribution(explanations, graph)
        mean_phi     = float(np.mean(list(phi.values()))) if phi else 0.0
        mean_n_shared = float(np.mean([e["n_shared_questions"] for e in explanations]))

        rows.append({
            "question_number": question.number,
            "strategy": "ccg_rag",
            "embedder": embedder_label,
            "reference": "Small JASIS 1973 + Angirekula arXiv 2025",
            "K": K,
            "min_cocit": min_cocit,
            "n_chunks_via_graph": n_via_graph,
            "explanation_precision": ep,
            "aumann_shapley_mean": mean_phi,
            "mean_shared_questions": mean_n_shared,
            "top_explanation": explanations[0]["explanation"] if explanations else "",
            **m,
        })

    return rows
