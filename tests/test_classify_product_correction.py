import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


# The production dependency is not required to unit-test the orchestration.
if "ollama" not in sys.modules:
    fake_ollama = ModuleType("ollama")
    fake_ollama.chat = lambda **kwargs: None
    sys.modules["ollama"] = fake_ollama

classify_module = importlib.import_module("src.classify_product")


def _response(code, status="found", description="description"):
    message = SimpleNamespace(
        content=json.dumps({
            "cn_code": code,
            "description": description,
            "status": status,
        }),
        tool_calls=[],
    )
    return SimpleNamespace(message=message)


def _tool_response():
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="unknown_tool", arguments={})
    )
    return SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[tool_call])
    )


def test_unknown_cn8_is_replaced_only_by_an_official_code(monkeypatch):
    responses = iter([
        _response(None, status="uncertain"),
        _response("12345678"),
        _response(None, status="uncertain"),
        _response("94039910"),
    ])
    monkeypatch.setattr(classify_module, "_chat", lambda **kwargs: next(responses))

    result = classify_module.classify_product(
        "product", valid_codes={"94039910"}
    )

    assert result["raw_cn_code"] == "12345678"
    assert result["raw_code_status"] == "Format valide mais code inconnu"
    assert result["cn_code"] == "94039910"
    assert result["code_status"] == "valid"
    assert result["correction_attempted"] is True
    assert result["correction_success"] is True
    assert result["correction_applied"] is True
    assert result["correction_method"] == "model_corrected_unknown_cn8"


def test_second_unknown_cn8_becomes_uncertain_without_hiding_raw(monkeypatch):
    responses = iter([
        _response(None, status="uncertain"),
        _response("12345678"),
        _response(None, status="uncertain"),
        _response("87654321"),
    ])
    monkeypatch.setattr(classify_module, "_chat", lambda **kwargs: next(responses))

    result = classify_module.classify_product(
        "product", valid_codes={"94039910"}
    )

    assert result["raw_cn_code"] == "12345678"
    assert result["cn_code"] is None
    assert result["status"] == "uncertain"
    assert result["code_status"] == "Il manque le code"
    assert result["correction_attempted"] is True
    assert result["correction_success"] is False
    assert result["correction_applied"] is False
    assert result["correction_method"] == "second_unknown_cn8_rejected"


def test_valid_raw_cn8_does_not_trigger_correction(monkeypatch):
    responses = iter([_response(None, status="uncertain"), _response("94039910")])
    monkeypatch.setattr(classify_module, "_chat", lambda **kwargs: next(responses))

    result = classify_module.classify_product(
        "product", valid_codes={"94039910"}
    )

    assert result["cn_code"] == "94039910"
    assert result["correction_attempted"] is False
    assert result["correction_success"] is False
    assert result["correction_applied"] is False


def test_unknown_code_correction_has_an_explicit_tool_limit(monkeypatch):
    monkeypatch.setattr(
        classify_module, "_chat", lambda **kwargs: _tool_response()
    )

    with pytest.raises(ValueError, match="pendant la correction"):
        classify_module._request_unknown_code_correction([], "12345678")


def test_tool_and_structured_requests_are_separate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        classify_module,
        "chat",
        lambda **kwargs: calls.append(kwargs) or _response(None, status="uncertain"),
    )

    classify_module._chat([], tools=classify_module.TOOLS)
    classify_module._chat([], structured=True)

    assert "tools" in calls[0]
    assert calls[0]["think"] is True
    assert "format" not in calls[0]
    assert "format" in calls[1]
    assert "tools" not in calls[1]


def test_textual_qwen_tool_call_is_recovered():
    message = SimpleNamespace(
        content=json.dumps({
            "name": "search_nomenclature_tool",
            "arguments": {"query": "grinding discs", "limit": 5},
        }),
        tool_calls=[],
    )

    calls = classify_module._tool_calls(message)

    assert calls[0].function.name == "search_nomenclature_tool"
    assert calls[0].function.arguments["query"] == "grinding discs"
