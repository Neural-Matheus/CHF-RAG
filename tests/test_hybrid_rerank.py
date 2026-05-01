from dataclasses import dataclass

from src.hybrid_rerank import BM25DocLevel, reciprocal_rank_fusion


@dataclass
class FakeChunk:

    doc_filename: str
    chunk_id: int
    text: str
    char_start: int = 0
    char_end: int = 0


def test_bm25_basic():

    chunks = [
        FakeChunk("doc_irpf.txt", 0,
                  "imposto de renda pessoa física dedução de despesas médicas"),
        FakeChunk("doc_csll.txt", 0,
                  "contribuição social sobre o lucro líquido alíquotas e bases"),
        FakeChunk("doc_pis.txt", 0,
                  "pis cofins regime cumulativo e não cumulativo"),
    ]
    bm25 = BM25DocLevel(chunks)
    top = bm25.retrieve("dedução de despesas médicas no imposto de renda", k=3)

    assert len(top) == 3
    assert top[0][0].doc_filename == "doc_irpf.txt"
    assert top[0][1] > top[1][1]


def test_bm25_doc_level_dedup():

    chunks = [
        FakeChunk("doc_a.txt", 0, "imposto"),
        FakeChunk("doc_a.txt", 1, "imposto renda"),
        FakeChunk("doc_a.txt", 2, "imposto renda dedução"),
        FakeChunk("doc_b.txt", 0, "lei tributária"),
        FakeChunk("doc_c.txt", 0, "outra coisa"),
    ]
    bm25 = BM25DocLevel(chunks)
    top = bm25.retrieve("imposto", k=10)
    docs = [c.doc_filename for c, _ in top]
    assert len(docs) == len(set(docs)), f"Duplicate docs: {docs}"
    assert "doc_a.txt" in docs


def test_bm25_chunks_vs_doclevel():

    chunks = [
        FakeChunk("doc_a.txt", 0, "imposto renda"),
        FakeChunk("doc_a.txt", 1, "imposto renda dedução"),
        FakeChunk("doc_a.txt", 2, "imposto renda crédito"),
    ]
    bm25 = BM25DocLevel(chunks)
    chunks_top = bm25.retrieve_chunks("imposto", k=3)
    docs_top = bm25.retrieve("imposto", k=3)
    assert len(chunks_top) == 3
    assert len(docs_top) == 1


def test_rrf_symmetric():

    chunks = [FakeChunk(f"doc_{i}.txt", 0, f"texto {i}") for i in range(5)]
    rank1 = [chunks[0], chunks[1], chunks[2], chunks[3]]
    rank2 = [chunks[0], chunks[1], chunks[2], chunks[3]]

    fused_ab = reciprocal_rank_fusion([rank1, rank2], k=60)
    fused_ba = reciprocal_rank_fusion([rank2, rank1], k=60)
    docs_ab = [c.doc_filename for c, _ in fused_ab]
    docs_ba = [c.doc_filename for c, _ in fused_ba]
    assert docs_ab == docs_ba
    assert docs_ab == ["doc_0.txt", "doc_1.txt", "doc_2.txt", "doc_3.txt"]


def test_rrf_consensus_wins():

    chunks = [FakeChunk(f"doc_{i}.txt", 0, "x") for i in range(5)]
    rank1 = [chunks[2], chunks[0], chunks[1]]
    rank2 = [chunks[2], chunks[1], chunks[3]]

    fused = reciprocal_rank_fusion([rank1, rank2], k=60)
    assert fused[0][0].doc_filename == "doc_2.txt"


def test_rrf_score_formula():

    chunks = [FakeChunk(f"doc_{i}.txt", 0, "x") for i in range(3)]
    rank = [chunks[0], chunks[1], chunks[2]]
    fused = reciprocal_rank_fusion([rank], k=60)
    expected = {
        "doc_0.txt": 1.0 / 61,
        "doc_1.txt": 1.0 / 62,
        "doc_2.txt": 1.0 / 63,
    }
    for c, s in fused:
        assert abs(s - expected[c.doc_filename]) < 1e-9
