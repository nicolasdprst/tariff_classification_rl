import json
from ollama import chat 
from search_nomenclature import search_nomenclature_tool

MODEL = "qwen3:1.7b"

messages = [
    {
        "role": "system",
        "content": "You are a customs classification assistant."
    },
    {
        "role": "user",
        "content": "Classify white chocolate containing no cocoa."
    }
]

response = chat(model=MODEL, messages=messages, tools=[search_nomenclature_tool])

for appel_tool in response.message.tool_calls:
    if appel_tool.function.name != "search_nomenclature_tool":
            raise ValueError(f"L'outil souhaité ({appel_tool.function.name}) n'existe pas")

    arg = appel_tool.function.arguments
    print(arg)
    result = search_nomenclature_tool(**arg)

    messages.append({
        "role" : "tool",
        "tool_name" : appel_tool.function.name,
        "content" : result
    })

print(response)


