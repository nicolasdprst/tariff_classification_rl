from classify_product import classify_product
import json

dataset_path = "data/validation_eu_en_tiny.jsonl"

def evaluate():
    results = {"id" : [], "product_description": [], "status": [], "predicted_cn8": [], "ground_truth_cn8": [], "correct": []}
    metrics = {"accuracy": 0, "uncertain": 0}
    
    for line in open(dataset_path):
        product = json.loads(line)
        product_description = product["product_description"]
        result = classify_product(product_description)
        results["id"].append(product["id"])
        results["product_description"].append(product_description)
        results["status"].append(result["status"])
        results["predicted_cn8"].append(result["cn_code"])
        results["ground_truth_cn8"].append(product["cn_code"])
        results["correct"].append(result["cn_code"]==product["cn_code"])

    metrics["accuracy"] = sum(results["correct"])/len(results["correct"])
    return results, metrics

if __name__ == "__main__":
    results, metrics = evaluate()
    print(f"Number of tests: {len(results['product_description'])}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")