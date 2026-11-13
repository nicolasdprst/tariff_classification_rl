import pandas as pd
from pathlib import Path

NOMENCLATURE_PATH = Path(__file__).resolve().parents[1] / "data" / "reference" / "nomenclature.xlsx"
df_nomenclature = pd.read_excel(NOMENCLATURE_PATH, header=4)  #3 colonnes : Texte, CN et Unit


def get_subcategories(code: str) -> list[dict]:
    """
    Retourne les sous-catégories directes d'un code tarifaire.
    """
    code = "".join(code.split())                                         #on enlève les espaces dans les codes
    if not code.isdigit() or len(code) not in {2, 4, 6, 8}:              #si le code ne fait pas la taille requise -> erreur
        raise ValueError("Le code doit contenir 2, 4, 6 ou 8 chiffres")
    if len(code) == 8:
        return []

    nomenclature = df_nomenclature[["CN", "Text"]].copy()               #on copie la nomenclature et on garde que le code et la description
    nomenclature["normalized_cn"] = (
        nomenclature["CN"].astype(str).str.replace(" ", "", regex=False)      #on enlève les espaces dans les codes
    )
    descendants = nomenclature[                                                #on trouve tous les descendants
        nomenclature["normalized_cn"].str.startswith(code)
        & (nomenclature["normalized_cn"].str.len() > len(code))
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

    
    
