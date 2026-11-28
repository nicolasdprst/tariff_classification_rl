"""On fait un cache rapide pour les appels chat et les embeddings pour éviter d'avoir à recacluler inutilement"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from ollama import Client


CACHE_DIR = Path(__file__).resolve().parents[1] / ".model_cache"
CLIENT = Client(timeout=180)


def _path(kind: str, payload: dict) -> Path: #le modèle et les paramètres font partie de la clé pour isoler les variantes
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    return CACHE_DIR / kind / f"{key}.json"


def cached_chat(**kwargs): #on rejoue une réponse identique si cet appel exact a déjà été effectué
    path = _path("chat", kwargs)
    if path.is_file():
        return SimpleNamespace(message=SimpleNamespace(content=json.loads(path.read_text(encoding="utf-8"))["content"]))
    response = CLIENT.chat(**kwargs)
    content = response.message.content or ""
    if content:
        #on évite de laisser une réponse partielle dans le cache
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    return response


def cached_embed(model: str, text: str) -> list[float]: #chaque texte a son propre cache réutilisé ensuite entre l'index et les requêtes
    payload = {"model": model, "prompt": text}
    path = _path("embed", payload)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["embedding"]
    result = CLIENT.embed(model=model, input=text)
    embedding = list(result.embeddings[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"embedding": embedding}), encoding="utf-8")
    temporary.replace(path)
    return embedding


def cached_embed_many(model: str, texts: list[str]) -> list[list[float]]: #on envoie que les textes absents du cache
    paths = [_path("embed", {"model": model, "prompt": text}) for text in texts]
    missing = [i for i, path in enumerate(paths) if not path.is_file()]
    for start in range(0, len(missing), 32):
        indices = missing[start:start + 32]
        result = CLIENT.embed(model=model, input=[texts[i] for i in indices])
        for index, embedding in zip(indices, result.embeddings):
            path = paths[index]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"embedding": list(embedding)}), encoding="utf-8")
            temporary.replace(path)
    return [json.loads(path.read_text(encoding="utf-8"))["embedding"] for path in paths]
