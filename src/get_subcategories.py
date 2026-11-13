import re

from src.search_nomenclature import df_nomenclature

normalized_nomenclature = df_nomenclature[["CN", "Text"]].copy()
normalized_nomenclature["normalized_cn"] = (
    normalized_nomenclature["CN"].astype(str).str.replace(" ", "", regex=False)
)


def get_subcategories(code: str) -> list[dict]:
    """
    Retourne les sous-catégories directes d'un code tarifaire.
    """
    if not isinstance(code, str):
        raise ValueError("Le code CN doit être une chaîne de caractères")

    # Les séparateurs d'affichage ne font pas partie du code douanier.
    code = re.sub(r"[\s.\-]", "", code)
    if code.isdigit() and len(code) == 10:
        raise ValueError(
            "Code TARIC à 10 chiffres reçu. Cet outil attend un code CN "
            "de 2, 4, 6 ou 8 chiffres ; choisissez un code CN à partir "
            "des résultats de recherche, sans inventer de subdivision."
        )
    if not code.isdigit() or len(code) not in {2, 4, 6, 8}:
        raise ValueError(
            "Le code CN doit contenir 2, 4, 6 ou 8 chiffres, "
            "hors espaces, points ou tirets"
        )
    if len(code) == 8:
        return []

    descendants = normalized_nomenclature[                              #on trouve tous les descendants
        normalized_nomenclature["normalized_cn"].str.startswith(code)
        & (normalized_nomenclature["normalized_cn"].str.len() > len(code))
    ]

    descendant_codes = set(descendants["normalized_cn"])
    direct_children = descendants[
        descendants["normalized_cn"].map(
            lambda candidate: not any(
                candidate.startswith(intermediate)
                and len(code) < len(intermediate) < len(candidate)
                for intermediate in descendant_codes
            )
        )
    ]

    return (
        direct_children[["CN", "Text"]]
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    
    
