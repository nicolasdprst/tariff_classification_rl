import re
from pathlib import Path
import pandas as pd 

NOMENCLATURE_PATH = Path(__file__).resolve().parents[1] / "data" / "reference" / "nomenclature.xlsx"
df_nomenclature = pd.read_excel(NOMENCLATURE_PATH, header=4)

normalized_nomenclature = df_nomenclature[["CN", "Text"]].copy()
normalized_nomenclature["normalized_cn"] = (
    normalized_nomenclature["CN"].astype(str).str.replace(" ", "", regex=False)
)


def get_subcategories(code: str) -> list[dict]:
    """Return the direct child categories of a 2-, 4-, or 6-digit CN code."""
    if not isinstance(code, str):
        raise ValueError("The CN code must be a string.")

    # Les séparateurs d'affichage ne font pas partie du code douanier.
    code = re.sub(r"[\s.\-]", "", code)
    if code.isdigit() and len(code) == 10:
        raise ValueError(
            "A 10-digit TARIC code was provided. This tool accepts only "
            "2-, 4-, or 6-digit CN parent codes. Select a CN code from the "
            "search results and do not invent a subdivision."
        )
    if not code.isdigit() or len(code) not in {2, 4, 6, 8}:
        raise ValueError(
            "The CN code must contain 2, 4, or 6 digits, excluding display "
            "spaces, dots, or hyphens."
        )
    if len(code) == 8:
        raise ValueError(
            f"Code {code} is already a terminal CN8 code and has no child "
            "categories. Stop using tools and return the final classification."
        )

    descendants = normalized_nomenclature[                              #on trouve tous les descendants
        normalized_nomenclature["normalized_cn"].str.startswith(code)
        & (normalized_nomenclature["normalized_cn"].str.len() > len(code))
    ]

    # sans descendant, il n'y a aucun enfant direct
    if descendants.empty:
        return []

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

    
    
