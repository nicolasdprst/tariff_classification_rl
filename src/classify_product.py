

import json
import re
from ollama import Client

from src.code_validation import normalize_customs_code, validate_cn8
from src.get_subcategories import get_subcategories
from src.get_valid_codes import get_valid_codes
from src.model_cache import cached_chat
from src.meilisearch.index_nomenclature import search_nomenclature_tool
from src.hybrid_retrieval import retrieve, clean_description
from src.nomenclature_tree import get_tree


MODEL = "qwen3:1.7b"
OLLAMA_TIMEOUT_SECONDS = 180
MAX_INITIAL_CANDIDATES = 20
NOMENCLATURE_HITS_PER_QUERY = 10
MAX_SELECTION_TOKENS = 128

CN_SECTIONS = [
    #sections pour les codes à 2 chiffres
    ("I", 1, 5, "Live animals and animal products"),
    ("II", 6, 14, "Vegetable products"),
    ("III", 15, 15, "Animal, vegetable or microbial fats and oils"),
    ("IV", 16, 24, "Prepared food, beverages and tobacco"),
    ("V", 25, 27, "Mineral products"),
    ("VI", 28, 38, "Chemical products"),
    ("VII", 39, 40, "Plastics and rubber"),
    ("VIII", 41, 43, "Hides, leather, furskins and articles thereof"),
    ("IX", 44, 46, "Wood, cork and basketware"),
    ("X", 47, 49, "Pulp, paper and printed products"),
    ("XI", 50, 63, "Textiles and textile articles"),
    ("XII", 64, 67, "Footwear, headgear, umbrellas and similar articles"),
    ("XIII", 68, 70, "Stone, ceramics and glass"),
    ("XIV", 71, 71, "Precious stones, precious metals and jewellery"),
    ("XV", 72, 83, "Base metals and articles of base metal"),
    ("XVI", 84, 85, "Machinery and electrical equipment"),
    ("XVII", 86, 89, "Vehicles and transport equipment"),
    ("XVIII", 90, 92, "Optical, measuring, medical and musical instruments"),
    ("XIX", 93, 93, "Arms and ammunition"),
    ("XX", 94, 96, "Furniture, toys and miscellaneous manufactured articles"),
    ("XXI", 97, 97, "Works of art, collectors' pieces and antiques"),
]

chat = cached_chat

SYSTEM_PROMPT = """
You are an EU customs classification assistant.

Your task is to classify products into the European Combined
Nomenclature (CN) by selecting the single best matching category from
the closed list of official CN options provided to you.

RULES
1. Choose the most appropriate code from OFFICIAL CANDIDATE CHOICES.
2. Match the goods by their actual nature, material, and function.
   Do not classify a consumable, part, or component as the machine that
   uses it unless the nomenclature classifies parts with the parent product.
3. Only return status "uncertain" with selected_code null if all listed
   categories are clearly unrelated to the product.
4. Return only the requested JSON object.
"""


def _selection_schema(allowed_codes: list[str]) -> dict:
    """Limite la réponse JSON aux codes proposés pour cette étape."""
    return {
        "type": "object",
        "properties": {
            "selected_code": {
                "anyOf": [
                    {"type": "string", "enum": allowed_codes},
                    {"type": "null"},
                ]
            },
            "status": {
                "type": "string",
                "enum": ["selected", "uncertain"],
            },
        },
        "required": ["selected_code", "status"],
        "additionalProperties": False,
    }


SELECTION_SCHEMA = _selection_schema([])

ABSTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "head_noun": {"type": "string"},
        "customs_nature": {"type": "string"},
        "material": {"type": "string"},
        "function": {"type": "string"},
        "physical_form": {
            "type": "string",
            "enum": [
                "complete_article", "machine", "part", "material",
                "preparation", "food", "other",
            ],
        },
        "parent_product": {"type": "string"},
        "search_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 2,
        },
    },
    "required": [
        "head_noun",
        "customs_nature",
        "material",
        "function",
        "physical_form",
        "parent_product",
        "search_queries",
    ],
    "additionalProperties": False,
}


def _chat_selection(messages: list[dict], allowed_codes: list[str] | None = None):
    """Demande à Qwen un choix court au format JSON contraint."""
    schema = (
        _selection_schema(allowed_codes)
        if allowed_codes
        else SELECTION_SCHEMA
    )
    return chat(
        model=MODEL,
        messages=messages,
        format=schema,
        think=False,
        options={
            "temperature": 0,
            "num_predict": MAX_SELECTION_TOKENS,
            "num_ctx": 8192,
        },
    )


def _abstract_product(description: str) -> dict:
    """Extrait la nature, la matière et des requêtes pour la recherche initiale."""
    prompt = (
        "Abstract this product into customs-search terminology. First identify "
        "the head noun: the simplest generic noun phrase stating what the item "
        "is. Do not include shape, colour, certifications, applications, users, "
        "or descriptive adjectives in head_noun. Then identify its material and "
        "primary function. Also identify its physical customs form: a complete "
        "article, machine, part, material, preparation, food, or other. If it "
        "is an identifiable part, state its parent product; otherwise use an "
        "empty string. Ignore brands, marketing language and dimensions. "
        "Use a broad commodity category likely to occur in a customs heading: "
        "for example 'dog food' rather than 'chewy treat', 'furniture part' "
        "rather than 'chrome-plated base', 'grinding wheel' rather than 'round "
        "disc', and 'culture media' rather than 'stem-cell medium'. "
        "Provide one or two short search queries of 2-5 generic commodity "
        "terms, built around the head noun. Each query must describe the goods "
        "themselves, not a machine that uses them or a vague class such as "
        "'chemical product', 'biological product', 'food', or 'mobility aid'. "
        "Prefer nomenclature wording over "
        "wording copied from the "
        f"description.\n\nPRODUCT DESCRIPTION:\n{description}"
    )
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=ABSTRACTION_SCHEMA,
        think=False,
        options={"temperature": 0, "num_predict": 256, "num_ctx": 8192},
    )
    try:
        #on vérifie les champs indispensables avant de lancer Meilisearch.
        profile = json.loads(response.message.content)
        head_noun = str(profile.get("head_noun", "")).strip()
        if not head_noun:
            raise ValueError("Missing head noun")
        profile["head_noun"] = head_noun
        queries = profile.get("search_queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError("Missing search queries")
        profile["search_queries"] = [
            str(query).strip() for query in queries[:2] if str(query).strip()
        ]
        if not profile["search_queries"]:
            raise ValueError("Empty search queries")
        return profile
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        # Si Qwen ne rend pas un profil exploitable, on cherche avec le texte brut.
        words = description.split()
        return {
            "head_noun": " ".join(words[:2]),
            "customs_nature": "unknown",
            "material": "unknown",
            "function": "unknown",
            "physical_form": "other",
            "parent_product": "",
            "search_queries": [" ".join(words[:5])],
        }


def _build_retrieval_queries(profile: dict) -> list[str]:
    """Prépare au plus trois requêtes distinctes à partir du profil produit."""
    head_noun = str(profile.get("head_noun", "")).strip()

    def usable(value: str) -> bool:
        return bool(value) and value.casefold() not in {"unknown", "none", "n/a"}

    #on évite d'être trop vague dans la requête sinon on se ramène à des résultats lexicaux hors sujet.
    candidates = [head_noun]
    candidates.extend(profile.get("search_queries") or [])

    queries = []
    seen = set()
    for candidate in candidates:
        query = " ".join(str(candidate).split())
        normalized = query.casefold()
        if usable(query) and normalized not in seen:
            seen.add(normalized)
            queries.append(query)
        if len(queries) == 3:
            break
    return queries


def _consolidate_searches(
    searches: list[dict], limit: int = MAX_INITIAL_CANDIDATES
) -> list[dict]:
    """Fusionne les réponses Meilisearch sans perdre les requêtes associées."""
    candidates: list[dict] = []
    by_code: dict[str, dict] = {}
    for search in searches:
        query = search["query"]
        for hit_rank, hit in enumerate(search["hits"]):
            normalized = "".join(
                character for character in str(hit.get("code", ""))
                if character.isdigit()
            )
            key = normalized or str(hit.get("code", ""))
            if key in by_code:
                if query not in by_code[key]["matched_queries"]:
                    by_code[key]["matched_queries"].append(query)
                by_code[key]["_best_hit_rank"] = min(
                    by_code[key]["_best_hit_rank"], hit_rank
                )
                continue
            candidate = {
                field: hit.get(field)
                for field in (
                    "code", "text", "hierarchy_text", "ancestor_codes", "unit"
                )
            }
            candidate["matched_queries"] = [query]
            candidate["_best_hit_rank"] = hit_rank
            candidate["_discovery_order"] = len(candidates)
            by_code[key] = candidate
            candidates.append(candidate)

    candidates.sort(key=lambda candidate: (
        #un code retrouvé par plusieurs requêtes passe avant un code isolé.
        -len(candidate["matched_queries"]),
        candidate["_best_hit_rank"],
        candidate["_discovery_order"],
    ))
    selected = candidates[:limit]
    for candidate in selected:
        candidate.pop("_best_hit_rank")
        candidate.pop("_discovery_order")
    return selected


def _candidate_codes(candidates: list[dict]) -> list[str]:
    """Récupère les codes distincts présents dans la recherche initiale."""
    codes = []
    seen = set()
    for candidate in candidates:
        code = "".join(
            character
            for character in str(candidate.get("code", ""))
            if character.isdigit()
        )
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _extract_candidate_chapters(candidates: list[dict]) -> list[dict]:
    """Regroupe les résultats initiaux par chapitre CN2."""
    chapter_hits: dict[str, list[str]] = {}
    for candidate in candidates:
        digits = "".join(
            character
            for character in str(candidate.get("code", ""))
            if character.isdigit()
        )
        if len(digits) < 2:
            continue
        chapter = digits[:2]
        label = str(
            candidate.get("hierarchy_text") or candidate.get("text") or ""
        ).strip()
        summary = label or f"Chapter {chapter}"
        bucket = chapter_hits.setdefault(chapter, [])
        if summary not in bucket and len(bucket) < 3:
            bucket.append(summary)

    return [
        {"code": chapter, "label": " | ".join(summaries)}
        for chapter, summaries in chapter_hits.items()
    ]


def _format_subcategory_options(subcategories: list[dict]) -> list[dict]:
    """Transforme les enfants de l'arbre en choix sans doublons pour Qwen."""
    options = []
    seen = set()
    for item in subcategories:
        raw_code = str(item.get("CN", ""))
        code = normalize_customs_code(raw_code)
        if not code or not code.isdigit() or code in seen:
            continue
        seen.add(code)
        options.append({
            "code": code,
            "label": str(item.get("Text", "")).strip(),
        })
    return options


def _supported_options(options: list[dict], candidates: list[dict]) -> list[dict]:
    """Repère les enfants dont le préfixe apparaît dans la recherche initiale."""
    candidate_codes = _candidate_codes(candidates)
    supported = [
        option
        for option in options
        if any(
            len(candidate) >= len(option["code"])
            and candidate.startswith(option["code"])
            for candidate in candidate_codes
        )
    ]
    return supported or options


def _section_options() -> list[dict]:
    return [
        {"code": section, "label": label}
        for section, _start, _end, label in CN_SECTIONS
    ]


def _chapter_options(section_code: str) -> list[dict]:
    """Présente chaque chapitre avec quelques positions CN4 représentatives."""
    section = next(
        (item for item in CN_SECTIONS if item[0] == section_code), None
    )
    if section is None:
        return []
    _name, start, end, _label = section
    options = []
    for chapter_number in range(start, end + 1):
        chapter = f"{chapter_number:02d}"
        headings = _format_subcategory_options(get_subcategories(chapter))
        representative = [
            heading["label"]
            for heading in headings
            if heading["label"].casefold() not in {"other", "others"}
        ][:4]
        if representative:
            options.append({
                "code": chapter,
                "label": "Examples of headings: " + " | ".join(representative),
            })
    return options


def _section_for_chapter(chapter: str) -> str | None:
    if not chapter.isdigit():
        return None
    number = int(chapter)
    return next(
        (
            section
            for section, start, end, _label in CN_SECTIONS
            if start <= number <= end
        ),
        None,
    )


def _evidence_options(
    options: list[dict], candidates: list[dict], level: str
) -> list[dict]:
    """Garde les sections ou chapitres soutenus par la recherche initiale."""
    candidate_codes = _candidate_codes(candidates)
    if level == "section":
        evidence = {
            _section_for_chapter(code[:2])
            for code in candidate_codes
            if len(code) >= 2
        }
    else:
        evidence = {code[:2] for code in candidate_codes if len(code) >= 2}
    return [option for option in options if option["code"] in evidence]


def _shortlist_options(
    options: list[dict],
    query: str,
    profile: dict,
    candidates: list[dict],
    limit: int = 18,
) -> list[dict]:
    """Réduit une longue liste d'enfants à partir des mots du produit et des libellés."""
    if len(options) <= limit:
        return options
    candidate_codes = _candidate_codes(candidates)
    supported_codes = {
        # On favorise les branches déjà retrouvées par Meilisearch.
        option["code"]
        for option in options
        if any(code.startswith(option["code"]) for code in candidate_codes)
    }
    search_text = " ".join([
        query,
        str(profile.get("head_noun", "")),
        str(profile.get("customs_nature", "")),
        str(profile.get("material", "")),
        str(profile.get("function", "")),
        " ".join(profile.get("search_queries") or []),
    ])
    def tokens(text: str) -> set[str]:
        normalized = set()
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            normalized.add(token)
        return normalized

    query_tokens = tokens(search_text)

    def score(option: dict) -> tuple[int, int]:
        label_tokens = tokens(option["label"])
        return (
            int(option["code"] in supported_codes),
            len(query_tokens & label_tokens),
        )

    return sorted(options, key=score, reverse=True)[:limit]


def _parse_selection(content: str, allowed_codes: set[str]) -> str | None:
    """Refuse un code absent des choix ou une réponse incertaine/invalide."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    selected_code = payload.get("selected_code")

    if status == "uncertain":
        return None
    if status != "selected":
        return None
    if not isinstance(selected_code, str) or selected_code not in allowed_codes:
        return None
    return selected_code


def _level_label(parent_code: str) -> str:
    """Nomme le niveau de nomenclature demandé à Qwen."""
    return {
        2: "CN4 heading (or direct sub-level)",
        4: "CN6 subheading (or direct CN8)",
        6: "CN8 tariff code",
    }.get(len(parent_code), "CN subcategory")


def _choose_category(
    query: str,
    profile: dict,
    level_name: str,
    options: list[dict],
    path_history: list[dict],
    evidence_candidates: list[dict] | None = None,
) -> dict | None:
    """Fait choisir une catégorie dans une liste fermée à chaque niveau."""
    if not options:
        return None
    if len(options) == 1:
        # Un seul enfant possible : aucun appel au modèle n'est nécessaire.
        return options[0]

    path_lines = "\n".join(
        f"- {step['code']}: {step['label']}" for step in path_history
    )
    evidence_candidates = evidence_candidates or []
    option_lines = "\n".join(
        f"- {option['code']}: {option['label']}" for option in options
    )
    path_section = f"SELECTED PATH SO FAR:\n{path_lines}\n\n" if path_lines else ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"PRODUCT DESCRIPTION:\n{query}\n\n"
                f"CUSTOMS ABSTRACTION:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
                f"{path_section}"
                f"CURRENT LEVEL: {level_name}\n"
                f"OFFICIAL CANDIDATE CHOICES:\n{option_lines}\n\n"
                "Pick the best matching code from OFFICIAL CANDIDATE CHOICES. "
                "Only use status \"uncertain\" "
                "with selected_code null if none of these categories can "
                "reasonably apply."
            ),
        },
    ]

    ordered_codes = [option["code"] for option in options]
    response = _chat_selection(messages, ordered_codes)
    content = getattr(getattr(response, "message", None), "content", "")
    allowed_codes = set(ordered_codes)
    selected_code = _parse_selection(content, allowed_codes)
    if selected_code is None:
        return None

    for option in options:
        if option["code"] == selected_code:
            return option
    return None


def _uncertain_result(
    initial_candidate_codes: list[str],
    valid_codes: set[str],
    raw_cn_code: str | None = None,
    raw_status: str = "uncertain",
    raw_code_status: str = "Il manque le code",
) -> dict:
    """Construit la réponse lorsque le parcours ne peut pas valider un CN8."""
    _, code_status = validate_cn8(None, valid_codes)
    return {
        "cn_code": None,
        "description": "",
        "status": "uncertain",
        "code_status": code_status,
        "raw_cn_code": raw_cn_code,
        "raw_status": raw_status,
        "raw_code_status": raw_code_status,
        "correction_attempted": False,
        "correction_success": False,
        "correction_applied": False,
        "correction_method": None,
        "initial_candidate_codes": initial_candidate_codes,
    }


def classify_product_rerank(query: str, valid_codes: set[str] | None = None,
                            product_id: str | None = None,
                            retrieval_mode: str = "lexical",
                            shuffle_seed: int | None = None,
                            think: bool = False) -> dict:
    """Choisit un seul CN8 parmi les candidats officiels avec un appel Qwen."""
    import random

    valid_codes = valid_codes if valid_codes is not None else get_valid_codes()
    candidates, neighbours = retrieve(
        query, k_nomenclature=35, k_train=5, product_id=product_id,
        embedding_mode=retrieval_mode,
    )
    candidates = [candidate for candidate in candidates if candidate["code"] in valid_codes][:40]
    #on nenvoie que des cn8 valides
    if not candidates:
        raise RuntimeError("Retriever returned no valid CN8 candidates")
    ranked_codes = [candidate["code"] for candidate in candidates]
    branch_scores = {}
    for candidate in candidates:
        prefix = candidate["code"][:4]
        branch_scores[prefix] = branch_scores.get(prefix, 0.0) + candidate["score"]
    best_branch = max(branch_scores, key=branch_scores.get)
    fallback = next(code for code in ranked_codes if code.startswith(best_branch))
    presented = candidates[:]
    if shuffle_seed is not None:
        #Le mélange sert à mesurer l'effet de l'ordre des options sur Qwen -> voir résultat qwen 1.7b
        random.Random(shuffle_seed).shuffle(presented)
    # Les voisins sont des exemples de réérence ; leurs descriptions sont tronquée.
    neighbour_lines = [
        f"{row['cn_code']}: {clean_description(row['product_description'])[:220]}"
        for row in neighbours[:3]
    ]
    options = "\n".join(f"{item['code']}: {item['hierarchy']}" for item in presented)
    schema = {
        # Le schéma contraint la réponse à l'un des codes effectivement proposé
        "type": "object",
        "properties": {"selected_code": {"type": "string", "enum": ranked_codes}},
        "required": ["selected_code"], "additionalProperties": False,
    }
    messages = [
        {"role": "system", "content": "Classify the goods under the EU Combined Nomenclature. Select exactly one offered CN8 code. Match the goods' nature, material and function; distinguish parts, consumables and complete machines. Return JSON only."},
        {"role": "user", "content": (
            f"GOODS:\n{clean_description(query)}\n\n"
            f"SIMILAR TRAINING RULINGS (examples, not answers):\n" + "\n".join(neighbour_lines) +
            f"\n\nOFFICIAL CN8 OPTIONS:\n{options}\n\nChoose the single best code."
        )},
    ]
    response = chat(model=MODEL, messages=messages, format=schema,
                    think=think, options={"temperature": 0,
                                          "num_predict": 512 if think else 128,
                                          "num_ctx": 8192})
    try:
        selected = json.loads(response.message.content)["selected_code"]
    except (ValueError, TypeError, KeyError, AttributeError):
        selected = fallback
    if selected not in ranked_codes:
        selected = fallback
    # On conserve les candidats présentés pour analyser ensuite les erreurs
    tree = get_tree()
    return {
        "cn_code": selected, "description": tree.labels[selected],
        "status": "found", "code_status": "valid", "raw_cn_code": selected,
        "raw_status": "found", "raw_code_status": "valid",
        "correction_attempted": False, "correction_success": False,
        "correction_applied": False, "correction_method": None,
        "initial_candidate_codes": ranked_codes,
        "candidate_details": presented,
    }


def classify_product(query: str, valid_codes: set[str] | None = None) -> dict:
    """Ancien parcours : section, chapitre puis enfants successifs jusqu'au CN8"""
    valid_codes = valid_codes if valid_codes is not None else get_valid_codes()
    # Qwen reformule le produit, puis Meilisearch cherche ses termes principau
    print("Building customs abstraction...", flush=True)
    profile = _abstract_product(query)
    retrieval_queries = _build_retrieval_queries(profile)
    initial_searches = []
    for search_query in retrieval_queries:
        initial_searches.append({
            "query": search_query,
            "hits": search_nomenclature_tool(
                search_query, limit=NOMENCLATURE_HITS_PER_QUERY
            ),
        })
    initial_candidates = _consolidate_searches(initial_searches)
    initial_candidate_codes = _candidate_codes(initial_candidates)
    print("Customs abstraction:", profile)
    print("Initial search results:", initial_searches)
    print("Consolidated candidates:", initial_candidates)

    section_options = _section_options()
    # Premier choix de section, puis second choix avec les indices de recherche
    section = _choose_category(
        query=query,
        profile=profile,
        level_name="CN section",
        options=section_options,
        path_history=[],
    )
    if section is None:
        return _uncertain_result(initial_candidate_codes, valid_codes)

    section_evidence = _evidence_options(
        section_options, initial_candidates, "section"
    )
    if section not in section_evidence:
        section_evidence.append(section)
    refined_section = _choose_category(
        query=query,
        profile=profile,
        level_name="CN section (final choice using search evidence)",
        options=section_evidence,
        path_history=[],
    )
    if refined_section is not None:
        section = refined_section

    chapter_options = _chapter_options(section["code"])
    # On applique le même principe au chapitre, dans la section sélectionnée.
    selected = _choose_category(
        query=query,
        profile=profile,
        level_name="CN chapter (2-digit)",
        options=chapter_options,
        path_history=[section],
    )
    if selected is None:
        return _uncertain_result(initial_candidate_codes, valid_codes)

    chapter_evidence = _evidence_options(
        chapter_options, initial_candidates, "chapter"
    )
    if selected not in chapter_evidence:
        chapter_evidence.append(selected)
    refined_chapter = _choose_category(
        query=query,
        profile=profile,
        level_name="CN chapter (final choice using search evidence)",
        options=chapter_evidence,
        path_history=[section],
    )
    if refined_chapter is not None:
        selected = refined_chapter

    path_history = [section, selected]
    current_code = selected["code"]
    last_allowed_codes = {opt["code"] for opt in chapter_options}

    while len(current_code) < 8:
        # Chaque réponse fixe le parent de l'étape suivante : pas de retour en arrière.
        subcategories = get_subcategories(current_code)
        options = _format_subcategory_options(subcategories)
        options = _shortlist_options(
            options, query, profile, initial_candidates
        )
        selected = _choose_category(
            query=query,
            profile=profile,
            level_name=_level_label(current_code),
            options=options,
            path_history=path_history,
            evidence_candidates=initial_candidates,
        )
        if selected is None:
            return _uncertain_result(initial_candidate_codes, valid_codes)
        path_history.append(selected)
        current_code = selected["code"]
        last_allowed_codes = {opt["code"] for opt in options}

    if len(current_code) != 8 or current_code not in last_allowed_codes:
        return _uncertain_result(initial_candidate_codes, valid_codes)

    # Le code terminal doit appartenir à la nomenclature officielle chargé
    normalized_code, code_status = validate_cn8(current_code, valid_codes)
    if code_status != "valid":
        return _uncertain_result(
            initial_candidate_codes,
            valid_codes,
            raw_cn_code=current_code,
            raw_status="found",
            raw_code_status=code_status,
        )

    return {
        "cn_code": normalized_code,
        "description": selected["label"],
        "status": "found",
        "code_status": code_status,
        "raw_cn_code": current_code,
        "raw_status": "found",
        "raw_code_status": code_status,
        "correction_attempted": False,
        "correction_success": False,
        "correction_applied": False,
        "correction_method": None,
        "initial_candidate_codes": initial_candidate_codes,
    }


if __name__ == "__main__":
    print(classify_product(query="muzzle-loading firearm"))
