"""Embedding avec bge_m3 pour retrouver les codes"""
import hashlib
from pathlib import Path

import numpy as np

from src.model_cache import CACHE_DIR, cached_embed, cached_embed_many
from src.nomenclature_tree import get_tree

MODEL = "bge-m3"
INDEX_PATH = CACHE_DIR / "official_cn_bge_m3.npz"


def _corpus():     # Il faut inclure le chemin hiérarchique et pas que le libellé final qui manque de contexte
    tree = get_tree()
    codes = sorted(code for code in tree.labels if len(code) in (4, 6, 8))
    texts = [tree.paths[code] for code in codes]
    digest = hashlib.sha256("\n".join(f"{c}\t{t}" for c, t in zip(codes, texts)).encode()).hexdigest()
    return codes, texts, digest


def build_index(batch_size=32):
    codes, texts, digest = _corpus()
    #si le modèle ou le contenu de la nomenclature change, on reconstruit l'index.
    if INDEX_PATH.exists():
        with np.load(INDEX_PATH) as saved:
            if str(saved["digest"]) == digest and str(saved["model"]) == MODEL:
                return
    vectors = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(cached_embed_many(MODEL, texts[start:start + batch_size]))
        if start % 320 == 0:
            print(f"Embedding official CN: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    matrix = np.asarray(vectors, dtype=np.float32)  
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = INDEX_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, codes=np.asarray(codes), matrix=matrix, digest=digest, model=MODEL)
    temporary.replace(INDEX_PATH)


_loaded = None


def search(text, limit=100):
    global _loaded
    if _loaded is None:
        build_index()
        with np.load(INDEX_PATH) as saved:
            _loaded = (saved["codes"].tolist(), saved["matrix"])
    codes, matrix = _loaded
    query = np.asarray(cached_embed(MODEL, text), dtype=np.float32)
    query /= max(np.linalg.norm(query), 1e-12)
    scores = matrix @ query              #on cherche directement dans tous les CN4, CN6 et CN8 indexés.
    count = min(limit, len(codes))
    indices = np.argpartition(-scores, count - 1)[:count]
    indices = indices[np.argsort(-scores[indices])]
    return [(codes[i], float(scores[i])) for i in indices]
