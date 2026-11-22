import json

from ollama import chat

from src.code_validation import get_valid_taric_prefix, validate_cn8
from src.get_subcategories import get_subcategories
from src.get_valid_codes import get_valid_codes
from src.meilisearch.index_nomenclature import search_nomenclature_tool


MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """
You are an EU customs classification assistant.

Your task is to classify products into the European Combined
Nomenclature (CN) using the available search tools.

Your final objective is to identify the correct 8-digit CN code.

AVAILABLE TOOLS

1. search_nomenclature_tool(query, limit=5)
   Searches the official CN nomenclature using keywords.

2. get_subcategories(code)
   Returns the direct child categories of a CN code.
   Use it only for codes with 2, 4, or 6 digits.
   Never call it with an 8-digit code.

SEARCH STRATEGY

1. Start by identifying the product's main nature,
   material, function, and relevant characteristics.
2. Search using short, distinctive keywords.
3. Prefer categories matching the product's function and nature.
4. If results are irrelevant, reformulate with different keywords.
5. Explore only a relevant parent code returned by a tool.
6. Never explore an unrelated code without evidence.

CN CLASSIFICATION RULES

1. CN codes have 2, 4, 6, then 8 hierarchical digits.
2. Select the most specific classification justified by the evidence.
3. A final code must explicitly appear in tool results.
4. Never invent a code or additional digits.
5. Never confuse 8-digit CN codes with 10-digit TARIC codes.
6. An empty result does not confirm that a code is valid.

TOOL USAGE RULES

1. Do not repeat an identical unsuccessful tool call.
2. Adapt after a tool error.
3. Prefer promising candidates over unrelated categories.
4. Minimize unnecessary calls and stop once a supported CN8 is found.

UNCERTAINTY

If the evidence does not support a specific CN8 code, do not guess.
Return an uncertain classification instead.

OUTPUT FORMAT

Return only the requested JSON object. Use an 8-digit CN code, or null when
uncertain. Do not include Markdown or explanations outside JSON.
"""

MAX_TOOL_CALLS = 5
MAX_CORRECTION_TOOL_CALLS = 3
TOOLS = [search_nomenclature_tool, get_subcategories]
TOOL_HANDLERS = {tool.__name__: tool for tool in TOOLS}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "cn_code": {
            "anyOf": [
                {"type": "string", "pattern": "^[0-9]{8}$"},
                {"type": "null"},
            ]
        },
        "description": {"type": "string"},
        "status": {"type": "string", "enum": ["found", "uncertain"]},
    },
    "required": ["cn_code", "description", "status"],
    "additionalProperties": False,
}


def _chat(messages: list[dict], tools=None):
    """Appeller Ollama avec une sortie controlée."""
    return chat(
        model=MODEL,
        messages=messages,
        tools=tools,
        format=OUTPUT_SCHEMA,
        options={"temperature": 0},
    )


def _parse_classification(content: str) -> dict:
    try:
        result = json.loads(content)
    except json.JSONDecodeError as err:
        raise ValueError(
            "Erreur de parsing JSON, le modèle n'a pas retourné un JSON "
            f"correct : {err}"
        ) from err

    required = {"cn_code", "description", "status"}
    if not isinstance(result, dict):
        raise ValueError("La réponse JSON doit être un objet")
    if not required.issubset(result):
        raise ValueError(f"Champs manquants : {required - result.keys()}")
    if result["status"] not in {"found", "uncertain"}:
        raise ValueError(f"Statut invalide : {result['status']}")
    if result["cn_code"] is not None and not isinstance(result["cn_code"], str):
        raise ValueError("cn_code doit être une chaîne ou null")
    if not isinstance(result["description"], str):
        raise ValueError("description doit être une chaîne")
    return result


def _request_taric_correction(messages: list, candidate: str) -> dict:
    correction_messages = messages + [{
        "role": "user",
        "content": (
            "FORMAT CORRECTION ONLY: you returned a potential 10-digit TARIC "
            f"code. Its official CN8 prefix is {candidate}. Return cn_code as "
            f"exactly \"{candidate}\" if this is the CN code you intended, "
            "otherwise return cn_code null with status \"uncertain\". Do not "
            "choose any other code."
        ),
    }]
    corrected = _parse_classification(
        _chat(messages=correction_messages).message.content
    )
    if corrected["cn_code"] not in {candidate, None}:
        raise ValueError("La correction a proposé un autre code")
    return corrected


def _request_unknown_code_correction(messages: list, invalid_code: str) -> dict:
    """Run one guided correction attempt, with a separate tool-call limit."""
    correction_messages = messages + [{
        "role": "user",
        "content": (
            f"CORRECTION REQUIRED: the proposed code {invalid_code} does not "
            "exist in the official 2026 Combined Nomenclature. Use the "
            "available search results and subcategories to identify an "
            "existing CN8 code. You may perform a new search or explore a "
            "relevant category. Never invent a subdivision and never complete "
            "a code with arbitrary zeros. If no reliable existing code can be "
            "identified, return cn_code null and status \"uncertain\"."
        ),
    }]
    response = _chat(messages=correction_messages, tools=TOOLS)
    correction_messages.append(response.message)

    tool_call_count = 0
    while response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            if tool_call_count >= MAX_CORRECTION_TOOL_CALLS:
                raise ValueError(
                    "Nombre d'appels d'outils dépassé pendant la correction "
                    f"({MAX_CORRECTION_TOOL_CALLS})"
                )
            tool_call_count += 1
            tool_name = tool_call.function.name
            tool_handler = TOOL_HANDLERS.get(tool_name)
            arguments = tool_call.function.arguments
            if tool_handler is None:
                tool_result = {
                    "error": f"Unknown tool: {tool_name}",
                    "invalid_arguments": arguments,
                }
            else:
                try:
                    tool_result = tool_handler(**arguments)
                except (TypeError, ValueError) as err:
                    tool_result = {
                        "error": str(err),
                        "invalid_arguments": arguments,
                    }
            correction_messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        response = _chat(messages=correction_messages, tools=TOOLS)
        correction_messages.append(response.message)
    return _parse_classification(response.message.content)


def classify_product(query: str, valid_codes: set[str] | None = None) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = _chat(messages=messages, tools=TOOLS)
    messages.append(response.message)

    tool_round = 0
    while response.message.tool_calls and tool_round < MAX_TOOL_CALLS:
        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            tool_handler = TOOL_HANDLERS.get(tool_name)
            if tool_handler is None:
                raise ValueError(f"L'outil souhaité ({tool_name}) n'existe pas")

            arguments = tool_call.function.arguments
            print("Arguments générés par Qwen:", arguments)
            try:
                tool_result = tool_handler(**arguments)
            except (TypeError, ValueError) as err:
                tool_result = {"error": str(err), "invalid_arguments": arguments}
            print("Résultats de la recherche:", tool_result)
            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        tool_round += 1
        response = _chat(messages=messages, tools=TOOLS)
        messages.append(response.message)

    if response.message.tool_calls:
        raise ValueError(
            f"Nombre de tours d'appels d'outils dépassé ({MAX_TOOL_CALLS})"
        )

    raw_result = _parse_classification(response.message.content)
    raw_code = raw_result["cn_code"]
    valid_codes = valid_codes if valid_codes is not None else get_valid_codes()
    raw_normalized, raw_code_status = validate_cn8(raw_code, valid_codes)

    result = dict(raw_result)
    result.update({
        "raw_cn_code": raw_code,
        "raw_status": raw_result["status"],
        "raw_code_status": raw_code_status,
        "correction_attempted": False,
        "correction_success": False,
        "correction_applied": False,
        "correction_method": None,
    })

    candidate = get_valid_taric_prefix(raw_code, valid_codes)
    if candidate is not None:
        result["correction_attempted"] = True
        try:
            corrected = _request_taric_correction(messages, candidate)
            corrected_code, corrected_status = validate_cn8(
                corrected["cn_code"], valid_codes
            )
            corrected["status"] = (
                "found" if corrected_status == "valid" else "uncertain"
            )
            result.update(corrected)
            result["cn_code"] = corrected_code
            result["correction_success"] = corrected_status == "valid"
            result["correction_applied"] = result["correction_success"]
            result["correction_method"] = (
                "model_confirmed_taric_cn8_prefix"
                if corrected_code == candidate
                else "model_rejected_taric_cn8_prefix"
            )
        except (ValueError, TypeError) as err:
            result["cn_code"] = raw_normalized
            result["correction_method"] = "taric_correction_failed"
            result["correction_error"] = str(err)
    elif raw_code_status == "Format valide mais code inconnu":
        result["correction_attempted"] = True
        try:
            corrected = _request_unknown_code_correction(
                messages, raw_normalized
            )
            corrected_code, corrected_status = validate_cn8(
                corrected["cn_code"], valid_codes
            )
            if corrected_status == "valid":
                result.update(corrected)
                result["cn_code"] = corrected_code
                result["status"] = "found"
                result["correction_success"] = True
                result["correction_applied"] = True
                result["correction_method"] = "model_corrected_unknown_cn8"
            else:
                result["cn_code"] = None
                result["description"] = ""
                result["status"] = "uncertain"
                result["correction_method"] = (
                    "model_uncertain_after_unknown_cn8"
                    if corrected["cn_code"] is None
                    else "second_unknown_cn8_rejected"
                )
        except Exception as err:
            result["cn_code"] = None
            result["description"] = ""
            result["status"] = "uncertain"
            result["correction_method"] = "unknown_cn8_correction_failed"
            result["correction_error"] = str(err)
    else:
        result["cn_code"] = raw_normalized
        if (
            isinstance(raw_normalized, str)
            and len(raw_normalized) == 10
            and raw_normalized.isdigit()
        ):
            result["correction_method"] = "taric_prefix_not_in_nomenclature"

    _, result["code_status"] = validate_cn8(result["cn_code"], valid_codes)
    return result


if __name__ == "__main__":
    print(classify_product(query="muzzle-loading firearm"))
