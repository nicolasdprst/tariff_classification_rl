import io
import json
import threading
import time

import evaluate as evaluate_module


def _classification(code):
    return {
        "raw_cn_code": code,
        "cn_code": code,
        "raw_status": "found",
        "status": "found",
        "code_status": "valid",
        "correction_attempted": False,
        "correction_success": False,
        "correction_applied": False,
        "correction_method": None,
    }


def test_evaluate_runs_batches_concurrently_and_preserves_order(monkeypatch):
    products = [
        {"id": str(index), "product_description": str(index), "cn_code": "12345678"}
        for index in range(6)
    ]
    dataset_content = "".join(json.dumps(product) + "\n" for product in products)

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def fake_classify(query, valid_codes):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return _classification("12345678")

    monkeypatch.setattr(evaluate_module, "get_valid_codes", lambda: {"12345678"})
    monkeypatch.setattr(evaluate_module, "classify_product", fake_classify)
    monkeypatch.setattr(
        "builtins.open", lambda *args, **kwargs: io.StringIO(dataset_content)
    )

    results, metrics = evaluate_module.evaluate(
        "validation.jsonl", batch_size=3, max_workers=3
    )

    assert state["peak"] == 3
    assert results["id"] == ["0", "1", "2", "3", "4", "5"]
    assert metrics["accuracy"] == 1.0


def test_evaluate_rejects_invalid_batch_configuration():
    for keyword in ("batch_size", "max_workers"):
        arguments = {"batch_size": 1, "max_workers": 1, keyword: 0}
        try:
            evaluate_module.evaluate("unused.jsonl", **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{keyword}=0 should be rejected")


def test_evaluate_retries_then_records_the_error(monkeypatch):
    product = {"id": "broken", "product_description": "x", "cn_code": "12345678"}
    monkeypatch.setattr(
        evaluate_module, "classify_product", lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("invalid JSON")
        )
    )
    monkeypatch.setattr(evaluate_module.time, "sleep", lambda delay: None)

    row = evaluate_module._evaluate_product_with_retries(
        product, {"12345678"}, max_retries=2
    )

    assert row["status"] == "error"
    assert row["evaluation_attempts"] == 3
    assert "invalid JSON" in row["evaluation_error"]


def test_empty_checkpoint_is_ignored(monkeypatch):
    product = {"id": "1", "product_description": "x", "cn_code": "12345678"}
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(json.dumps(product)))
    monkeypatch.setattr(evaluate_module.Path, "is_file", lambda path: True)
    monkeypatch.setattr(
        evaluate_module.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(evaluate_module.pd.errors.EmptyDataError),
    )
    monkeypatch.setattr(evaluate_module, "_save_checkpoint", lambda *args: None)
    monkeypatch.setattr(evaluate_module, "get_valid_codes", lambda: {"12345678"})
    monkeypatch.setattr(
        evaluate_module,
        "classify_product",
        lambda *args, **kwargs: _classification("12345678"),
    )

    results, _ = evaluate_module.evaluate(
        "validation.jsonl", checkpoint_path="results.csv", max_retries=0
    )

    assert results["id"] == ["1"]

