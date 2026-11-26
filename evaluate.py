import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import pandas as pd

from src.classify_product import classify_product
from src.code_validation import validate_cn8
from src.get_valid_codes import get_valid_codes
from src.reward import calculate_reward


DATASET_PATH = "data/reference/validation_eu_en_tiny.jsonl"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_RETRIES = 0
RESULT_COLUMNS = [
    "id", "product_description", "raw_status", "status", "raw_predicted",
    "predicted_cn8", "ground_truth_cn8", "raw_code_status", "code_status",
    "correction_attempted", "correction_success", "correction_applied",
    "correction_method", "raw_correct", "correct", "raw_reward", "reward",
    "evaluation_attempts", "evaluation_error",
]


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate_product(product: dict, valid_codes: set[str]) -> dict:
    """Classify one product and build its evaluation row."""
    classification = classify_product(
        product["product_description"], valid_codes=valid_codes
    )
    raw_code = classification["raw_cn_code"]
    raw_normalized, raw_code_status = validate_cn8(raw_code, valid_codes)
    predicted_code = classification["cn_code"]
    code_status = classification["code_status"]

    raw_correct = (
        raw_code_status == "valid" and raw_normalized == product["cn_code"]
    )
    corrected_correct = (
        code_status == "valid" and predicted_code == product["cn_code"]
    )

    return {
        "id": product["id"],
        "product_description": product["product_description"],
        "raw_status": classification["raw_status"],
        "status": classification["status"],
        "raw_predicted": raw_code,
        "predicted_cn8": predicted_code,
        "ground_truth_cn8": product["cn_code"],
        "raw_code_status": raw_code_status,
        "code_status": code_status,
        "correction_attempted": classification["correction_attempted"],
        "correction_success": classification["correction_success"],
        "correction_applied": classification["correction_applied"],
        "correction_method": classification["correction_method"],
        "raw_correct": raw_correct,
        "correct": corrected_correct,
        "raw_reward": calculate_reward(
            raw_normalized, product["cn_code"], valid_codes
        ),
        "reward": calculate_reward(
            predicted_code, product["cn_code"], valid_codes
        ),
        "evaluation_attempts": 1,
        "evaluation_error": None,
    }


def _failed_row(product: dict, attempts: int, error: Exception) -> dict:
    """Keep an evaluation usable even if one product repeatedly fails."""
    return {
        "id": product["id"],
        "product_description": product["product_description"],
        "raw_status": "error",
        "status": "error",
        "raw_predicted": None,
        "predicted_cn8": None,
        "ground_truth_cn8": product["cn_code"],
        "raw_code_status": "evaluation_error",
        "code_status": "evaluation_error",
        "correction_attempted": False,
        "correction_success": False,
        "correction_applied": False,
        "correction_method": None,
        "raw_correct": False,
        "correct": False,
        "raw_reward": 0.0,
        "reward": 0.0,
        "evaluation_attempts": attempts,
        "evaluation_error": f"{type(error).__name__}: {error}",
    }


def _evaluate_product_with_retries(
    product: dict, valid_codes: set[str], max_retries: int
) -> dict:
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            row = _evaluate_product(product, valid_codes)
            row["evaluation_attempts"] = attempt
            return row
        except Exception as error:
            if attempt == attempts:
                return _failed_row(product, attempt, error)
            # A short stagger prevents every failed worker from retrying at once.
            time.sleep(0.5 * attempt)


def _batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _save_checkpoint(results: dict, checkpoint_path: str) -> None:
    """Atomically replace the checkpoint so interruption cannot empty it."""
    destination = Path(checkpoint_path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    pd.DataFrame(results).to_csv(temporary, index=False)
    temporary.replace(destination)


def evaluate(
    dataset_path: str = DATASET_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    checkpoint_path: str | None = None,
    resume: bool = True,
):
    if batch_size < 1:
        raise ValueError("batch_size doit etre superieur ou egal a 1")
    if max_workers < 1:
        raise ValueError("max_workers doit etre superieur ou egal a 1")
    if max_retries < 0:
        raise ValueError("max_retries doit etre superieur ou egal a 0")

    results = {column: [] for column in RESULT_COLUMNS}
    valid_codes = get_valid_codes()

    with open(dataset_path, encoding="utf-8") as dataset:
        products = [json.loads(line) for line in dataset if line.strip()]

    completed_ids = set()
    if resume and checkpoint_path and Path(checkpoint_path).is_file():
        try:
            checkpoint = pd.read_csv(checkpoint_path, dtype={"id": str})
        except pd.errors.EmptyDataError:
            checkpoint = pd.DataFrame(columns=RESULT_COLUMNS)
            print("Checkpoint vide ignore: nouvelle evaluation")
        checkpoint = checkpoint.reindex(columns=RESULT_COLUMNS)
        dataset_ids = {str(product["id"]) for product in products}
        checkpoint = checkpoint[checkpoint["id"].isin(dataset_ids)]
        for record in checkpoint.to_dict(orient="records"):
            for key in RESULT_COLUMNS:
                value = record[key]
                results[key].append(None if pd.isna(value) else value)
            completed_ids.add(str(record["id"]))
        if completed_ids:
            print(f"Reprise: {len(completed_ids)}/{len(products)} produits deja evalues")

    pending_products = [
        product for product in products if str(product["id"]) not in completed_ids
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        completed = len(completed_ids)
        for batch in _batched(pending_products, batch_size):
            rows = executor.map(
                lambda product: _evaluate_product_with_retries(
                    product, valid_codes, max_retries
                ),
                batch,
            )
            for row in rows:
                for key, value in row.items():
                    results[key].append(value)
            completed += len(batch)
            if checkpoint_path:
                _save_checkpoint(results, checkpoint_path)
            print(f"Evaluation: {completed}/{len(products)} produits termines")

    metrics = {
        "raw_accuracy": _rate(results["raw_correct"]),
        "corrected_accuracy": _rate(results["correct"]),
        "raw_invalid_rate": _rate([
            status != "valid" for status in results["raw_code_status"]
        ]),
        "corrected_invalid_rate": _rate([
            status != "valid" for status in results["code_status"]
        ]),
        "raw_mean_reward": (
            sum(results["raw_reward"]) / len(results["raw_reward"])
            if results["raw_reward"] else 0.0
        ),
        "mean_reward": (
            sum(results["reward"]) / len(results["reward"])
            if results["reward"] else 0.0
        ),
        "raw_code_status_counts": dict(Counter(results["raw_code_status"])),
        "code_status_counts": dict(Counter(results["code_status"])),
        "correction_attempt_count": sum(results["correction_attempted"]),
        "correction_success_count": sum(results["correction_success"]),
        "evaluation_error_count": sum(
            error is not None for error in results["evaluation_error"]
        ),
    }
    # Compatibility with consumers of the former, single-path metrics.
    metrics["accuracy"] = metrics["corrected_accuracy"]
    metrics["invalid_rate"] = metrics["corrected_invalid_rate"]
    return results, metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CN classifications")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--output", default="evaluation_results.csv")
    parser.add_argument(
        "--no-resume", action="store_true", help="Ignore an existing output CSV"
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    results, metrics = evaluate(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        checkpoint_path=args.output,
        resume=not args.no_resume,
    )
    print(f"Number of tests: {len(results['product_description'])}")
    print(f"Raw accuracy: {metrics['raw_accuracy']:.4f}")
    print(f"Corrected accuracy: {metrics['corrected_accuracy']:.4f}")
    print(f"Raw mean reward: {metrics['raw_mean_reward']:.4f}")
    print(f"Corrected mean reward: {metrics['mean_reward']:.4f}")
    print(f"Raw invalid rate: {metrics['raw_invalid_rate']:.2%}")
    print(f"Corrected invalid rate: {metrics['corrected_invalid_rate']:.2%}")
    print(f"Correction attempts: {metrics['correction_attempt_count']}")
    print(f"Successful corrections: {metrics['correction_success_count']}")
    print(f"Evaluation errors: {metrics['evaluation_error_count']}")
    _save_checkpoint(results, args.output)
