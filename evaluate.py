from src.classify_product import classify_product
from src.reward import calculate_reward
from src.get_valid_codes import get_valid_codes
import json

dataset_path = "data/validation_eu_en_tiny.jsonl"

def evaluate():
    results = {"id" : [], "product_description": [], "status": [], "predicted_cn8": [], "ground_truth_cn8": [], "correct": [], "reward" : []}
    metrics = {"accuracy": 0, "uncertain": 0, "mean_reward" : 0}
    valid_codes = get_valid_codes()
    
    for line in open(dataset_path):
        product = json.loads(line)
        product_description = product["product_description"]
        result = classify_product(product_description)

        reward = calculate_reward(
            predicted_code=result["cn_code"],
            true_code=product["cn_code"],
            valid_codes=valid_codes
        )

        results["id"].append(product["id"])
        results["product_description"].append(product_description)
        results["status"].append(result["status"])
        results["predicted_cn8"].append(result["cn_code"])
        results["ground_truth_cn8"].append(product["cn_code"])
        results["correct"].append(result["cn_code"]==product["cn_code"])
        results["reward"].append(reward)

    metrics["accuracy"] = sum(results["correct"])/len(results["correct"])
    metrics["mean_reward"] = sum(results["reward"]) / len(results["reward"])
    return results, metrics

if __name__ == "__main__":
    results, metrics = evaluate()
    print(f"Number of tests: {len(results['product_description'])}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean reward: {metrics['mean_reward']:.4f}")
