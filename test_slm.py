import json
from ollama import chat 
from search_nomenclature import search_nomenclature_tool

MODEL = "qwen3:1.7b"

messages = [
    {
        "role": "system",
        "content": """You are a customs classification assistant. 
        When calling search_nomenclature_tool:
        - Use short, relevant product keywords.
        - Avoid copying the entire user description.
        - Prefer the main product name.
        - An empty search result does not mean that
        the product has no customs classification.
        """
    },
    {
        "role": "user",
        "content": "Classify white chocolate containing no cocoa."
    }
]

response = chat(model=MODEL, messages=messages, tools=[search_nomenclature_tool])

messages.append(response.message)
if response.message.tool_calls :  #on vérifie si c'est vide ou non
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

final_response = chat(model=MODEL,messages=messages)

print(final_response.message)


