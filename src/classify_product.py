from src.get_subcategories import get_subcategories
import json
import re
from ollama import chat 
from src.search_nomenclature import search_nomenclature_tool

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
   Avoid copying the entire product description.

3. Examine the returned candidates carefully.
   Prefer categories matching the product's function
   and nature, not merely individual keywords.

4. If the results are irrelevant, reformulate the
   search using different meaningful keywords.

5. Use get_subcategories only to explore a relevant
   parent code previously returned by a tool.

6. Never explore an unrelated code without evidence
   from the search results.

CN CLASSIFICATION RULES

1. CN codes have a hierarchical structure:
   - 2 digits: Chapter
   - 4 digits: Heading
   - 6 digits: HS subheading
   - 8 digits: CN subheading

2. Select the most specific classification justified
   by the product description and tool results.

3. A final 8-digit code must explicitly appear
   in the results returned by an available tool.

4. Never invent a CN code or additional digits.

5. Never confuse CN codes (8 digits) with
   TARIC codes (10 digits).

6. An empty result from get_subcategories does not
   confirm that a code exists or is valid.

TOOL USAGE RULES

1. Do not repeat an identical tool call if it has
   already failed or returned no useful results.

2. If a tool returns an error, adapt your approach
   instead of repeating the same arguments.

3. Prefer exploring promising candidates over
   searching unrelated categories.

4. Minimize unnecessary tool calls.

5. If a valid and well-supported 8-digit code has
   already been found, stop searching.

UNCERTAINTY

If the available evidence does not support a
specific 8-digit CN code, do not guess.

Return an uncertain classification instead.

OUTPUT FORMAT

Return ONLY a valid JSON object:

{
    "cn_code": "12345678",
    "description": "Official CN description",
    "status": "found"
}

If uncertain, return:

{
    "cn_code": null,
    "description": "",
    "status": "uncertain"
}

Do not include Markdown or explanations outside JSON.
"""


MAX_TOOL_CALLS = 5
TOOLS = [search_nomenclature_tool, get_subcategories]
TOOL_HANDLERS = {
    tool.__name__: tool
    for tool in TOOLS
}

def classify_product(query: str) -> dict:
    round = 0       
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": query
        }
    ]

    response = chat(model=MODEL, messages=messages, tools=TOOLS)

    messages.append(response.message)
    while response.message.tool_calls and round < MAX_TOOL_CALLS :  #on vérifie si c'est vide ou non
        for appel_tool in response.message.tool_calls:
            tool_name = appel_tool.function.name
            tool_handler = TOOL_HANDLERS.get(tool_name)
            if tool_handler is None:
                raise ValueError(f"L'outil souhaité ({appel_tool.function.name}) n'existe pas")

            arg = appel_tool.function.arguments
            print("Arguments générés par Qwen:", arg)
            try:
                result = tool_handler(**arg)
            except (TypeError, ValueError) as err:
                # Erreur rencontré notamment : le SLM cherche un code à 10chiffres car le prompt peut l'influencer
                # à chercher un code TARIC (8 + 2) et non un code CN (sans les deux derniers), on le précise donc
                # au SLM au lieu de juste tronqué sa réponse qui n'est pas forcément correcte
                result = {
                    "error": str(err),
                    "invalid_arguments": arg,
                }
            print("Résultats de la recherche:", result)

            messages.append({
                "role" : "tool",
                "tool_name" : appel_tool.function.name,
                "content" : json.dumps(result, ensure_ascii=False)
            })
        round+=1
        response = chat(model=MODEL, messages=messages, tools=TOOLS)
        messages.append(response.message)
    
    if response.message.tool_calls:             #si on a depassé la limite, on  informe
        raise ValueError(f"Nombre de tours d'appels d'outils dépassé ({MAX_TOOL_CALLS})")
    try:
        result = json.loads(response.message.content)
    except json.JSONDecodeError as err:
        raise ValueError(f"Erreur de parsing JSON, le modèle n'a pas retourné un JSON correct : {err}")
    
    required_fields = {"cn_code", "description", "status"}

    if not required_fields.issubset(result):
        missing_fields = required_fields - result.keys()
        raise ValueError(f"Champs manquants : {missing_fields}")

    if result["status"] not in {"found", "uncertain"}:
        raise ValueError(
            f"Statut invalide : {result['status']}"
        )

    if result["cn_code"] is not None and not isinstance(
        result["cn_code"], str
    ):
        raise ValueError("cn_code doit être une chaîne ou null")

    if not isinstance(result["description"], str):
        raise ValueError("description doit être une chaîne")

    if result["cn_code"] is not None:
        # La validité est évaluée ensuite par validate_cn8. Une mauvaise
        # prédiction est ainsi comptée comme invalide sans arrêter tout le lot.
        result["cn_code"] = re.sub(r"[\s.\-]", "", result["cn_code"])

    return result

if __name__ == "__main__":
    result = classify_product(query="muzzle-loading firearm")
    print("Query : muzzle-loading firearm")
    print(result)

