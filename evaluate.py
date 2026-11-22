from collections import Counter
import json

import pandas as pd

from src.classify_product import classify_product
from src.code_validation import validate_cn8
from src.get_valid_codes import get_valid_codes
from src.reward import calculate_reward


DATASET_PATH = "data/cleaned/validation_eu_en_tiny.jsonl"


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(dataset_path: str = DATASET_PATH):
    results = {
        "id": [],
        "product_description": [],
        "raw_status": [],
        "status": [],
        "raw_predicted": [],
        "predicted_cn8": [],
        "ground_truth_cn8": [],
        "raw_code_status": [],
        "code_status": [],
        "correction_attempted": [],
        "correction_success": [],
        "correction_applied": [],
        "correction_method": [],
        "raw_correct": [],
        "correct": [],
        "raw_reward": [],
        "reward": [],
    }
    valid_codes = get_valid_codes()

    with open(dataset_path, encoding="utf-8") as dataset:
        for line in dataset:
            product = json.loads(line)
            classification = classify_product(
                product["product_description"], valid_codes=valid_codes
            )
            raw_code = classification["raw_cn_code"]
            raw_normalized, raw_code_status = validate_cn8(raw_code, valid_codes)
            predicted_code = classification["cn_code"]
            code_status = classification["code_status"]

            raw_correct = (
                raw_code_status == "valid"
                and raw_normalized == product["cn_code"]
            )
            corrected_correct = (
                code_status == "valid"
                and predicted_code == product["cn_code"]
            )

            row = {
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
            }
            for key, value in row.items():
                results[key].append(value)

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
    }
    # Compatibility with consumers of the former, single-path metrics.
    metrics["accuracy"] = metrics["corrected_accuracy"]
    metrics["invalid_rate"] = metrics["corrected_invalid_rate"]
    return results, metrics


if __name__ == "__main__":
    results, metrics = evaluate()
    print(f"Number of tests: {len(results['product_description'])}")
    print(f"Raw accuracy: {metrics['raw_accuracy']:.4f}")
    print(f"Corrected accuracy: {metrics['corrected_accuracy']:.4f}")
    print(f"Raw mean reward: {metrics['raw_mean_reward']:.4f}")
    print(f"Corrected mean reward: {metrics['mean_reward']:.4f}")
    print(f"Raw invalid rate: {metrics['raw_invalid_rate']:.2%}")
    print(f"Corrected invalid rate: {metrics['corrected_invalid_rate']:.2%}")
    print(f"Correction attempts: {metrics['correction_attempt_count']}")
    print(f"Successful corrections: {metrics['correction_success_count']}")
    pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)
