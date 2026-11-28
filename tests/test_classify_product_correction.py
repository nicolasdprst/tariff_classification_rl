import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


# The production dependency is not required to unit-test the orchestration.
if "ollama" not in sys.modules:
    fake_ollama = ModuleType("ollama")
    fake_ollama.Client = lambda **kwargs: SimpleNamespace(
        chat=lambda **chat_kwargs: None
    )
    sys.modules["ollama"] = fake_ollama

classify_module = importlib.import_module("src.classify_product")


@pytest.fixture(autouse=True)
def stub_initial_retrieval(monkeypatch):
    monkeypatch.setattr(
        classify_module,
        "_abstract_product",
        lambda description: {
            "head_noun": "product",
            "customs_nature": "product",
            "material": "unknown",
            "function": "unknown",
            "physical_form": "other",
            "parent_product": "",
            "search_queries": ["product"],
        },
    )
    monkeypatch.setattr(
        classify_module, "search_nomenclature_tool", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        classify_module,
        "_section_options",
        lambda: [{"code": "TEST", "label": "Test section"}],
    )
    monkeypatch.setattr(
        classify_module,
        "_chapter_options",
        lambda section: classify_module._extract_candidate_chapters(
            classify_module._consolidate_searches([{
                "query": "test",
                "hits": classify_module.search_nomenclature_tool("test"),
            }])
        ),
    )
    monkeypatch.setattr(
        classify_module, "_evidence_options", lambda *args, **kwargs: []
    )


def _selection_response(code, status="selected"):
    message = SimpleNamespace(
        content=json.dumps({
            "selected_code": code,
            "status": status,
        })
    )
    return SimpleNamespace(message=message)


def test_hierarchical_navigation_selects_chapter_cn4_cn6_cn8(monkeypatch):
    monkeypatch.setattr(
        classify_module,
        "search_nomenclature_tool",
        lambda *args, **kwargs: [
            {"code": "8703 10 18", "text": "Electric vehicles", "hierarchy_text": "Vehicles > Motor cars"},
            {"code": "8427 20 19", "text": "Fork-lift trucks", "hierarchy_text": "Machinery > Trucks"},
        ],
    )
    tree = {
        "87": [
            {"CN": "8701", "Text": "Tractors"},
            {"CN": "8703", "Text": "Motor cars"},
        ],
        "8703": [
            {"CN": "8703 10", "Text": "Golf cars and similar"},
            {"CN": "8703 21", "Text": "Spark-ignition <= 1000 cm3"},
        ],
        "870310": [
            {"CN": "8703 10 11", "Text": "For travelling on snow"},
            {"CN": "8703 10 18", "Text": "Other"},
        ],
    }
    monkeypatch.setattr(
        classify_module, "get_subcategories", lambda code: tree[code]
    )
    responses = iter([
        _selection_response("87"),
        _selection_response("8703"),
        _selection_response("870310"),
        _selection_response("87031018"),
    ])
    monkeypatch.setattr(
        classify_module, "_chat_selection", lambda messages, allowed=None: next(responses)
    )

    result = classify_module.classify_product(
        "electric wheelchair", valid_codes={"87031018"}
    )

    assert result["cn_code"] == "87031018"
    assert result["raw_cn_code"] == "87031018"
    assert result["status"] == "found"
    assert result["code_status"] == "valid"
    assert result["correction_attempted"] is False
    assert result["initial_candidate_codes"] == ["87031018", "84272019"]


def test_singleton_levels_are_auto_selected_without_calling_qwen(monkeypatch):
    monkeypatch.setattr(
        classify_module,
        "search_nomenclature_tool",
        lambda *args, **kwargs: [
            {"code": "9403 99 10", "text": "Of metal"},
        ],
    )
    tree = {
        "94": [{"CN": "9403", "Text": "Other furniture and parts thereof"}],
        "9403": [{"CN": "9403 99", "Text": "Other parts"}],
        "940399": [{"CN": "9403 99 10", "Text": "Of metal"}],
    }
    monkeypatch.setattr(
        classify_module, "get_subcategories", lambda code: tree[code]
    )
    calls = []
    monkeypatch.setattr(
        classify_module,
        "_chat_selection",
        lambda messages, allowed=None: calls.append(messages) or _selection_response(None, "uncertain"),
    )

    result = classify_module.classify_product(
        "metal furniture base", valid_codes={"94039910"}
    )

    assert calls == []
    assert result["cn_code"] == "94039910"
    assert result["status"] == "found"


def test_direct_cn4_to_cn8_transition_is_supported(monkeypatch):
    monkeypatch.setattr(
        classify_module,
        "search_nomenclature_tool",
        lambda *args, **kwargs: [
            {"code": "0101 30 00", "text": "Asses"},
        ],
    )
    tree = {
        "01": [
            {"CN": "0101", "Text": "Live horses, asses, mules and hinnies"},
            {"CN": "0102", "Text": "Live bovine animals"},
        ],
        "0101": [
            {"CN": "0101 29", "Text": "Other"},
            {"CN": "0101 30 00", "Text": "Asses"},
        ],
    }
    monkeypatch.setattr(
        classify_module, "get_subcategories", lambda code: tree[code]
    )
    responses = iter([
        _selection_response("0101"),
        _selection_response("01013000"),
    ])
    monkeypatch.setattr(
        classify_module, "_chat_selection", lambda messages, allowed=None: next(responses)
    )

    result = classify_module.classify_product(
        "live donkey", valid_codes={"01013000"}
    )

    assert result["cn_code"] == "01013000"
    assert result["status"] == "found"


def test_out_of_list_code_returns_uncertain(monkeypatch):
    monkeypatch.setattr(
        classify_module,
        "search_nomenclature_tool",
        lambda *args, **kwargs: [
            {"code": "8703 10 18", "text": "Vehicles"},
            {"code": "8427 20 19", "text": "Trucks"},
        ],
    )
    monkeypatch.setattr(
        classify_module,
        "_chat_selection",
        lambda messages, allowed=None: _selection_response("99"),
    )

    result = classify_module.classify_product(
        "product", valid_codes={"87031018"}
    )

    assert result["cn_code"] is None
    assert result["status"] == "uncertain"
    assert result["code_status"] == "Il manque le code"


def test_invalid_json_returns_uncertain(monkeypatch):
    monkeypatch.setattr(
        classify_module,
        "search_nomenclature_tool",
        lambda *args, **kwargs: [
            {"code": "8703 10 18", "text": "Vehicles"},
            {"code": "8427 20 19", "text": "Trucks"},
        ],
    )
    monkeypatch.setattr(
        classify_module,
        "_chat_selection",
        lambda messages, allowed=None: SimpleNamespace(
            message=SimpleNamespace(content="not valid json")
        ),
    )

    result = classify_module.classify_product(
        "product", valid_codes={"87031018"}
    )

    assert result["cn_code"] is None
    assert result["status"] == "uncertain"


def test_uncertain_choice_stops_navigation(monkeypatch):
    monkeypatch.setattr(
        classify_module,
        "search_nomenclature_tool",
        lambda *args, **kwargs: [
            {"code": "8703 10 18", "text": "Vehicles"},
            {"code": "8427 20 19", "text": "Trucks"},
        ],
    )
    monkeypatch.setattr(
        classify_module,
        "_chat_selection",
        lambda messages, allowed=None: _selection_response(None, status="uncertain"),
    )

    result = classify_module.classify_product(
        "product", valid_codes={"87031018"}
    )

    assert result["cn_code"] is None
    assert result["status"] == "uncertain"


def test_empty_initial_candidates_returns_uncertain_without_calling_qwen(monkeypatch):
    calls = []
    monkeypatch.setattr(
        classify_module,
        "_chat_selection",
        lambda messages, allowed=None: calls.append(messages),
    )

    result = classify_module.classify_product(
        "product", valid_codes={"87031018"}
    )

    assert calls == []
    assert result["cn_code"] is None
    assert result["status"] == "uncertain"
    assert result["initial_candidate_codes"] == []


def test_chat_selection_uses_constrained_settings_without_tools(monkeypatch):
    calls = []
    monkeypatch.setattr(
        classify_module,
        "chat",
        lambda **kwargs: calls.append(kwargs) or _selection_response(None, "uncertain"),
    )

    classify_module._chat_selection(
        [{"role": "user", "content": "choose"}], ["87", "96"]
    )

    assert len(calls) == 1
    assert "tools" not in calls[0]
    assert calls[0]["think"] is False
    assert calls[0]["options"]["temperature"] == 0
    assert calls[0]["options"]["num_predict"] == classify_module.MAX_SELECTION_TOKENS
    assert calls[0]["format"]["properties"]["selected_code"]["anyOf"][0]["enum"] == [
        "87", "96"
    ]


def test_retrieval_queries_are_limited_and_exclude_broad_nature_queries():
    queries = classify_module._build_retrieval_queries({
        "head_noun": "rack",
        "customs_nature": "furniture",
        "material": "wood",
        "search_queries": ["wooden boot rack", "furniture for footwear"],
    })

    assert queries == ["rack", "wooden boot rack", "furniture for footwear"]


def test_retrieval_queries_ignore_unknowns_and_duplicates():
    queries = classify_module._build_retrieval_queries({
        "head_noun": "Culture   media",
        "customs_nature": "culture media",
        "material": "unknown",
        "search_queries": ["CULTURE MEDIA", "stem cell culture media"],
    })

    assert queries == ["Culture media", "stem cell culture media"]


def test_consolidated_candidates_deduplicate_codes_and_keep_query_evidence():
    candidates = classify_module._consolidate_searches([
        {
            "query": "copper foil",
            "hits": [{
                "code": "7410 11 00",
                "text": "Of refined copper",
                "hierarchy_text": "Copper foil > Of refined copper",
                "ancestor_codes": ["7410"],
                "unit": "-",
            }],
        },
        {
            "query": "electronic copper foil",
            "hits": [{
                "code": "7410 11 00",
                "text": "Of refined copper",
                "hierarchy_text": "Copper foil > Of refined copper",
                "ancestor_codes": ["7410"],
                "unit": "-",
            }],
        },
    ])

    assert len(candidates) == 1
    assert candidates[0]["matched_queries"] == [
        "copper foil", "electronic copper foil"
    ]


def test_consolidated_candidates_keep_top_hits_from_later_queries():
    searches = [
        {
            "query": f"query {query_index}",
            "hits": [
                {"code": f"{query_index}{hit_index:07d}", "text": "candidate"}
                for hit_index in range(5)
            ],
        }
        for query_index in range(3)
    ]

    candidates = classify_module._consolidate_searches(searches)

    assert len(candidates) == min(
        15, classify_module.MAX_INITIAL_CANDIDATES
    )
    assert {candidate["code"] for candidate in candidates[:3]} == {
        "00000000", "10000000", "20000000"
    }


def test_candidate_codes_are_normalized_and_deduplicated():
    assert classify_module._candidate_codes([
        {"code": "74 10 11 00"},
        {"code": "74101100"},
        {"code": "74.10"},
        {"code": None},
    ]) == ["74101100", "7410"]


def test_supported_options_prune_unretrieved_branches():
    options = [
        {"code": "9401", "label": "Seats"},
        {"code": "9403", "label": "Other furniture"},
        {"code": "9405", "label": "Lamps"},
    ]
    candidates = [
        {"code": "9403 60 90"},
        {"code": "9401 61 00"},
    ]

    assert classify_module._supported_options(options, candidates) == options[:2]


def test_supported_options_fall_back_when_no_candidate_reaches_child_level():
    options = [
        {"code": "9401", "label": "Seats"},
        {"code": "9403", "label": "Other furniture"},
    ]

    assert classify_module._supported_options(options, [{"code": "94"}]) == options


def test_long_official_option_lists_are_shortlisted_without_training_data():
    options = [
        {"code": f"84{index:02d}", "label": f"Machine category {index}"}
        for index in range(30)
    ]
    options[20]["label"] = "Machines for grinding metals"

    shortlisted = classify_module._shortlist_options(
        options,
        "metal grinding machine",
        {"head_noun": "grinding machine", "search_queries": []},
        [],
        limit=10,
    )

    assert len(shortlisted) == 10
    assert options[20] in shortlisted


