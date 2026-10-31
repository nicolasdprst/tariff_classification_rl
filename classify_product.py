import json
from ollama import chat 
from search_nomenclature import search_nomenclature_tool

MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """
You are an EU customs classification assistant.

Your task is to classify products using the EU Combined Nomenclature (CN).

SEARCH RULES:
1. Use search_nomenclature_tool to find relevant CN codes.
2. Search using short product keywords, not full descriptions.
3. Use the tool results as your primary source of evidence.
4. An empty search result does not mean the product has no classification.

CLASSIFICATION RULES:
1. Select the most specific CN code supported by the product
   description and the tool results.
2. Prefer an 8-digit CN code when sufficient information is available.
3. Never invent a CN code.
4. Do not confuse general headings with specific subheadings.
5. If the available information is insufficient, indicate uncertainty.

CN HIERARCHY RULES:

1. CN codes follow a hierarchical structure:
   - 2 digits: Chapter
   - 4 digits: Heading
   - 6 digits: HS subheading
   - 8 digits: CN subheading

2. A general heading must not be selected when a more
   specific subheading matches the product description.

3. Compare all retrieved candidates before selecting a code.

4. Prefer the deepest matching code, provided the
   product description contains enough information
   to justify that classification.

OUTPUT RULES:
Return only a JSON object with the following fields:
- cn_code: the selected CN code, or null if unknown.
- description: the corresponding official nomenclature description.
- status: "found" or "uncertain".

Do not include explanations outside the JSON object.
"""

MAX_TOOL_CALLS = 5

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

    response = chat(model=MODEL, messages=messages, tools=[search_nomenclature_tool])

    messages.append(response.message)
    while response.message.tool_calls and round < MAX_TOOL_CALLS :  #on vérifie si c'est vide ou non
        for appel_tool in response.message.tool_calls:
            if appel_tool.function.name != "search_nomenclature_tool":
                raise ValueError(f"L'outil souhaité ({appel_tool.function.name}) n'existe pas")

            arg = appel_tool.function.arguments
            print("Arguments générés par Qwen:", arg)
            result = search_nomenclature_tool(**arg)
            print("Résultats de la recherche:", result)

            messages.append({
                "role" : "tool",
                "tool_name" : appel_tool.function.name,
                "content" : json.dumps(result, ensure_ascii=False)
            })
        round+=1
        response = chat(model=MODEL, messages=messages, tools=[search_nomenclature_tool])
        messages.append(response.message)
    
    if response.message.tool_calls:             #si on a depassé la limite, on  informe
        raise ValueError(f"nombre d'appels à 'search_nomenclature_tool' dépassé ({MAX_TOOL_CALLS})")
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

    return result

if __name__ == "__main__":
    result = classify_product(query="muzzle-loading firearm")
    print(result)

