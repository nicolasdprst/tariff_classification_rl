"""Retrieval hybrid : on assemble tout ce que l'on avait déjà mis en place -> Meilisearch, TD-IDF et on ajoute l'embedding et les voisins trains"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.nomenclature_tree import get_tree
from src.meilisearch.index_nomenclature import search_nomenclature_tool
from src.model_cache import cached_embed, cached_embed_many
from src.embedding_index import search as semantic_search


ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "bge-m3"                      #On a utilisé bge-m3 pour embed, voir ici : https://huggingface.co/BAAI/bge-m3
# et voir le embedding_index.py

def clean_description(text: str) -> str:
    """Supprime les amorces administratives avant de chercher le produit."""
    return re.sub(
        r"^\s*(?:Applicant(?:'s)? (?:information|description|details):|"
        r"According to (?:the )?(?:application|information provided)[,:]?|"
        r"Description:|Details:|Information:|The product described as)",
        "", text, flags=re.I,
    ).strip()


@lru_cache(maxsize=1)
def _index():                                     # on construit une seule fois TF-IDF pour éviter de recharger à chaque fois
    tree = get_tree()
    leaves = sorted(code for code in tree.labels if len(code) == 8)
    train = [json.loads(line) for line in (ROOT / "data/reference/train_eu_en.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    matrix = vectorizer.fit_transform([tree.paths[code] for code in leaves])
    train_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    train_matrix = train_vectorizer.fit_transform([row["product_description"] for row in train])
    return tree, leaves, train, vectorizer, matrix, train_vectorizer, train_matrix


def _cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@lru_cache(maxsize=1)
def _train_embedding_matrix():                 #pareil, on calcule puis on réutilise
    train = _index()[2]
    matrix = np.asarray(cached_embed_many(EMBED_MODEL, [row["product_description"] for row in train]), dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix


def retrieve(description: str, k_nomenclature: int = 40, k_train: int = 5,
             product_id: str | None = None, use_embeddings: bool = True,
             embedding_mode: str = "lexical"):
    """Fusionne les recherches officielles et les exemples train en candidats CN8."""
    tree, leaves, train, vectorizer, matrix, train_vectorizer, train_matrix = _index()
    text = clean_description(description)
    # 1 -> similarité lexicale avec toutes les feuilles officielles.
    tfidf_scores = linear_kernel(vectorizer.transform([text]), matrix).ravel()
    lexical = sorted(range(len(leaves)), key=lambda i: tfidf_scores[i], reverse=True)[:300]
    train_scores = linear_kernel(train_vectorizer.transform([text]), train_matrix).ravel()
    # L'identifiant empêche qu'un exemple évalué serve de son propre voisin.
    neighbours = [train[i] for i in sorted(range(len(train)), key=lambda i: train_scores[i], reverse=True)
                  if train[i]["id"] != product_id][:k_train]
    rankings = [[leaves[i] for i in lexical]]
    try:
        #meilisearch apporte une deuxième liste et on évite de bloquer TF-IDFD
        meili = search_nomenclature_tool(text, limit=100)
        rankings.append([re.sub(r"\s+", "", hit["code"]) for hit in meili
                         if len(re.sub(r"\s+", "", hit["code"])) == 8])
    except Exception:
        rankings.append([])
    if use_embeddings and embedding_mode == "full":
        # Ici bge-m3 interroge l'index complet, sans présélection lexicale.
        semantic_hits = semantic_search(text, limit=300)
        leaf_rank = [code for code, _ in semantic_hits if len(code) == 8]
        # Une position CN4/CN6 pertinente peut faire remonter ses feuilles CN8.
        for parent, _ in semantic_hits:
            if len(parent) in (4, 6):
                leaf_rank.extend(code for code in leaves if code.startswith(parent))
        rankings.append(list(dict.fromkeys(leaf_rank)))
        query_embedding = np.asarray(cached_embed(EMBED_MODEL, text), dtype=np.float32)
        query_embedding /= max(np.linalg.norm(query_embedding), 1e-12)
        train_scores_semantic = _train_embedding_matrix() @ query_embedding
        semantic_train = np.argsort(-train_scores_semantic)
        semantic_neighbours = [train[i] for i in semantic_train if train[i]["id"] != product_id][:k_train]
        for row in semantic_neighbours:
            if row["id"] not in {neighbour["id"] for neighbour in neighbours}:
                neighbours.append(row)
    elif use_embeddings:
        #bge-m3 ne reranke que 300 feuilles TF-IDF
        embeddings = cached_embed_many(EMBED_MODEL, [text] + [tree.paths[leaves[i]] for i in lexical])
        query_embedding = embeddings[0]
        semantic = sorted(range(len(lexical)), key=lambda j: _cosine(query_embedding, embeddings[j + 1]), reverse=True)
        rankings.append([leaves[lexical[j]] for j in semantic])
    scores = {}
    #fusion RRF -> chaque liste apporte un score selon le rang du code.
    for ranking in rankings:
        for rank, code in enumerate(ranking):
            scores[code] = scores.get(code, 0.0) + 1 / (60 + rank + 1)
    for rank, row in enumerate(neighbours):
        code = row["cn_code"]
        if code in tree.paths:
            scores[code] = scores.get(code, 0.0) + 1 / (60 + rank + 1)
    branch_totals = {} #un CN8 profite légèrement des autres feuilles retrouvées dans son CN4
    for code, value in scores.items():
        branch_totals[code[:4]] = branch_totals.get(code[:4], 0.0) + value
    for code, value in list(scores.items()):
        scores[code] = value + 0.1 * (branch_totals[code[:4]] - value)
    if embedding_mode == "full" and use_embeddings:
        #on réserve quelques places aux voisins et aux résultats lexicaux avant le tri fusionné.
        lexical_seed = [row["cn_code"] for row in neighbours[:k_train] if row["cn_code"] in tree.paths]
        lexical_seed.extend(leaves[i] for i in lexical[:20])
        lexical_seed = list(dict.fromkeys(lexical_seed))[:20]
        semantic_train_seed = [row["cn_code"] for row in neighbours[k_train:k_train + 3]
                               if row["cn_code"] in tree.paths]
        codes = list(dict.fromkeys(lexical_seed + semantic_train_seed +
                                   sorted(scores, key=scores.get, reverse=True)))[:k_nomenclature + k_train]
    else:
        seed = [row["cn_code"] for row in neighbours if row["cn_code"] in tree.paths]
        seed.extend(leaves[i] for i in lexical[:20])
        codes = list(dict.fromkeys(seed + sorted(scores, key=scores.get, reverse=True)))[:k_nomenclature + k_train]
    return ([{"code": code, "hierarchy": tree.paths[code], "score": scores[code]} for code in codes], neighbours)
