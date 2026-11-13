import re


def validate_cn8(predicted_code: str | None, valid_codes: set[str]) -> tuple[str | None, str]:
    """fonction qui permet de s'assurer que le code proposé par le SLM est correct"""

    if predicted_code is None:
        return None, "Il manque le code"

    if not isinstance(predicted_code, str):
        return None, "Format invalide"

    code = re.sub(r"[\s.\-]", "", predicted_code)          #on enlève les séparateurs d'affichage

    if not code:           
        return None, "Il manque le code"

    if re.fullmatch(r"(?:[0-9]{2}|[0-9]{4}|[0-9]{6})", code):      #on a que le début du code (sur-classe)
        return code, "Réponse incomplète"

    if not re.fullmatch(r"[0-9]{8}", code):  #si le format est différent de 8 chiffres
        return code, "Format invalide"

    if code not in valid_codes:              #si le code est à un format correct mais n'étant pas dans la liste des codes existants
        return code, "Format valide mais code inconnu"

    return code, "Code valide"
