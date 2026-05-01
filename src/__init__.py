from .data_loader import load_questions, load_legal_docs, Question, LegalChunk
from .embedder import Embedder, EMBEDDER_CONFIGS

from .baselines import (
    ir_metrics,
    ir_metrics_at_ks,
    evaluate_cosine,
    evaluate_cosine_doc_level,
)

from .ccg_rag import build_cocitation_graph

from .heatflow_metrics import (
    erc_star,
    auc_f1,
    aggregate_t_star,
    energy_along_trajectory,
    empirical_decay_rate,
    cheeger_bound_check,
    spectral_bound_check,
    flow_signatures_for_topk,
    provenance_entropy,
    coupling_lift,
    coupling_work_ratio,
    operator_diagnostics,
    FlowSignature,
    TStarStats,
)

from .question_typing import (
    classify_question_type,
    classify_question_by_refs,
    classify_question_by_ref_count,
    count_diploma_mentions,
    stratify_t_star,
    StratifiedTStarStats,
)

from .coupled_heatflow import (
    classify_doc_hierarchy,
    diagnose_hierarchy_classification,
    build_semantic_laplacian,
    build_cocitation_laplacian,
    build_hierarchy_laplacian,
    build_coupled_operator,
    initial_distribution,
    score_at_time,
    channel_flow,
    retrieve_at_time,
    search_optimal_t,
    CoupledOperator,
)

from .hybrid_rerank import (
    BM25DocLevel,
    reciprocal_rank_fusion,
    CrossEncoderReranker,
    evaluate_bm25_pure,
    evaluate_with_rerank,
)
