"""Permet  de retourner les enfants des codes à 2,4 ou 6 chiffres sans avoir à rescanner le excel"""
import re
from src.nomenclature_tree import get_tree


def get_subcategories(code: str) -> list[dict]:
    """Retourne les enfants directs d'un chapitre, d'un CN4 ou d'un CN6."""
    if not isinstance(code, str):
        raise ValueError("The CN code must be a string.")

    code = re.sub(r"[\s.\-]", "", code)    #les séparateur ne font pas partie du code douanier.
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
    if len(code) == 8:                          #Si c'est un code terminal, on le précise au SLM (tjr en anglais pour pas mélanger les
        raise ValueError(                       #langues dans son contexte)
            f"Code {code} is already a terminal CN8 code and has no child "
            "categories. Stop using tools and return the final classification."
        )

    return get_tree().subcategories(code)

    
    
