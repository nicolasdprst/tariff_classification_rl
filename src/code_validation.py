import re


def normalize_customs_code(predicted_code: str) -> str:
    """Enlever les séparateurs sans changer les digits"""
    return re.sub(r"[\s.\-]", "", predicted_code)


def validate_cn8(
    predicted_code: str | None, valid_codes: set[str]
) -> tuple[str | None, str]:
    """normaliser le code de sortie en fonction de si c'est un bon code CN8 ou non"""
    if predicted_code is None:
        return None, "Il manque le code"
    if not isinstance(predicted_code, str):
        return None, "Format invalide"

    code = normalize_customs_code(predicted_code)
    if not code:
        return None, "Il manque le code"
    if re.fullmatch(r"(?:[0-9]{2}|[0-9]{4}|[0-9]{6})", code):
        return code, "Réponse incomplète"
    if not re.fullmatch(r"[0-9]{8}", code):
        return code, "Format invalide"
    if code not in valid_codes:
        return code, "Format valide mais code inconnu"
    return code, "valid"


def get_valid_taric_prefix(
    predicted_code: str | None, valid_codes: set[str]
) -> str | None:
    """Return the official CN8 prefix of a potential TARIC code.

    Returning a candidate does not apply a correction. The caller must
    explicitly confirm it, for example through a constrained model retry.
    """
    if not isinstance(predicted_code, str):
        return None
    code = normalize_customs_code(predicted_code)
    if not re.fullmatch(r"[0-9]{10}", code):
        return None
    candidate = code[:8]
    return candidate if candidate in valid_codes else None
