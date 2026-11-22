"""Indexation et recherche de la nomenclature douaniere avec Meilisearch

Exemples:
    python -m src.meilisearch.index_nomenclature index
    python -m src.meilisearch.index_nomenclature search "white chocolate"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import meilisearch
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOMENCLATURE_PATH = PROJECT_ROOT / "data" / "reference" / "nomenclature.xlsx"
DEFAULT_URL = "http://localhost:7700"
DEFAULT_INDEX = "nomenclature"

INDEX_SETTINGS: dict[str, Any] = {
    "searchableAttributes": ["text", "code", "code_normalized"],
    "displayedAttributes": ["id", "code", "text", "unit"],
    "filterableAttributes": ["has_code"],
    # Une faute de frappe acceptee dans un code douanier peut etre trompeuse.
    "typoTolerance": {"disableOnAttributes": ["code", "code_normalized"]},
}


def get_client(url: str | None = None, api_key: str | None = None) -> meilisearch.Client:
    """Construit un client depuis les arguments ou les variables d'environnement."""

    return meilisearch.Client(
        url or os.getenv("MEILISEARCH_URL", DEFAULT_URL),
        api_key if api_key is not None else os.getenv("MEILISEARCH_API_KEY"),
    )


def _clean_cell(value: Any) -> str:
    """Convertit une cellule Excel sans transformer NaN en ``"nan"``."""

    return "" if pd.isna(value) else str(value).strip()


def load_documents(path: Path = NOMENCLATURE_PATH) -> list[dict[str, Any]]:
    """Charge le classeur et produit les documents attendus par meilisearch"""

    dataframe = pd.read_excel(path, header=4, dtype=object)
    required_columns = {"CN", "Text", "Unit"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colonnes absentes du fichier de nomenclature : {missing}")

    documents: list[dict[str, Any]] = []
    for row_number, row in dataframe.iterrows():
        code = _clean_cell(row["CN"])
        text = _clean_cell(row["Text"])
        unit = _clean_cell(row["Unit"])
        if not code and not text and not unit:
            continue

        documents.append(
            {
                "id": f"cn_{row_number + 1:05d}",
                "code": code,
                "code_normalized": re.sub(r"\s+", "", code),
                "text": text,
                "unit": unit,
                "has_code": bool(code),
            }
        )

    return documents


def _wait_for_task(client: meilisearch.Client, task: Any, timeout_ms: int) -> None:
    completed = client.wait_for_task(task.task_uid, timeout_in_ms=timeout_ms)
    if completed.status == "failed":
        error = getattr(completed, "error", None)
        raise RuntimeError(f"La tache Meilisearch {task.task_uid} a echoué et : {error}")


def index_nomenclature(
    client: meilisearch.Client | None = None,
    index_uid: str | None = None,
    path: Path = NOMENCLATURE_PATH,
    timeout_ms: int = 120_000,
) -> int:
    """Configure l'index, ajoute les documents et retourne leur nombre."""

    client = client or get_client()
    index = client.index(index_uid or os.getenv("MEILISEARCH_INDEX", DEFAULT_INDEX))
    documents = load_documents(path)

    _wait_for_task(client, index.update_settings(INDEX_SETTINGS), timeout_ms)
    _wait_for_task(
        client, index.add_documents(documents, primary_key="id"), timeout_ms
    )
    return len(documents)


def search_nomenclature(
    query: str,
    limit: int = 5,
    client: meilisearch.Client | None = None,
    index_uid: str | None = None,
) -> list[dict[str, Any]]:
    """Recherche un libelle ou un code CN et retourne les resultats."""

    if limit < 1:
        raise ValueError("limit doit etre superieur ou egal a 1")

    client = client or get_client()
    index = client.index(index_uid or os.getenv("MEILISEARCH_INDEX", DEFAULT_INDEX))
    response = index.search(
        query.strip(),
        {
            "limit": limit,
            # Les intertitres du classeur aident a l'indexation, mais ne sont
            # pas des classifications retournables puisqu'ils n'ont pas de CN.
            "filter": "has_code = true",
            "attributesToRetrieve": ["code", "text", "unit"],
        },
    )
    return response["hits"]


def search_nomenclature_tool(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Tool Ollama : recherche des candidats CN dans l'index Meilisearch"""

    return search_nomenclature(query=query, limit=limit)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Indexer le classeur Excel")
    index_parser.add_argument("--file", type=Path, default=NOMENCLATURE_PATH)

    search_parser = subparsers.add_parser("search", help="Rechercher dans l'index")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "index":
            count = index_nomenclature(path=args.file)
            uid = os.getenv("MEILISEARCH_INDEX", DEFAULT_INDEX)
            print(f"{count} documents indexes dans '{uid}'.")
        else:
            hits = search_nomenclature(args.query, args.limit)
            print(json.dumps(hits, ensure_ascii=False, indent=2))
    except meilisearch.errors.MeilisearchError as error:
        raise SystemExit(f"Erreur Meilisearch : {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

