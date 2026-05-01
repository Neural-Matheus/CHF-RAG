from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


_E5_MODELS = {
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-small",
    "intfloat/e5-base",
    "intfloat/e5-large",
}


EMBEDDER_CONFIGS = {
    "intfloat/multilingual-e5-base": {
        "label": "E5-multilingual",
        "domain": "generic",
        "use_prefix": True,
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "label": "MiniLM-multilingual",
        "domain": "generic",
        "use_prefix": False,
    },
    "rufimelo/Legal-BERTimbau-sts-large-ma-v3": {
        "label": "Legal-BERTimbau",
        "domain": "legal-pt",
        "use_prefix": False,
    },
    "stjiris/bert-large-portuguese-cased-legal-mlm-sts-v1.0": {
        "label": "STJIRIS-Legal",
        "domain": "legal-pt",
        "use_prefix": False,
    },
}


class Embedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = None,
        batch_size: int = 64,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.use_prefix = model_name in _E5_MODELS
        self.config = EMBEDDER_CONFIGS.get(model_name, {"label": model_name, "domain": "unknown", "use_prefix": self.use_prefix})

        logger.info("Carregando embedder '%s' (%s) em %s ...",
                    self.config["label"], self.config["domain"], device)
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()
        logger.info("Embedder pronto. Dimensão: %d | Prefixo E5: %s",
                    self.dim, self.use_prefix)

    def embed(self, texts: List[str], prefix: str = "", show_progress: bool = True) -> np.ndarray:
        if self.use_prefix and prefix:
            texts = [f"{prefix}{t}" for t in texts]

        all_embeddings = []
        batches = range(0, len(texts), self.batch_size)
        if show_progress:
            batches = tqdm(batches, desc=f"Embedding [{self.config['label']}]", unit="batch")

        for start in batches:
            batch = texts[start: start + self.batch_size]
            emb = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(emb)

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_queries(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        return self.embed(texts, prefix="query: ", show_progress=show_progress)

    def embed_passages(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        return self.embed(texts, prefix="passage: ", show_progress=show_progress)

