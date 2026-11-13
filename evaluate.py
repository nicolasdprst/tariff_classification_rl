from src.classify_product import classify_product
from src.reward import calculate_reward
from src.get_valid_codes import get_valid_codes
from src.code_validation import validate_cn8
from collections import Counter
import json

dataset_path = "data/validation_eu_en_tiny.jsonl"

def evaluate():
    results = {"id" : [], "product_description": [], "status": [], "predicted_cn8": [], "ground_truth_cn8": [], "correct": [], "reward" : [], "raw_predicted": [],
                "code_status": []}
                # id : identifiant du produit
                # product_description : description du produit
                # status : statut retourné par SLM
                # raw_predicted_cn8 : code brut généré par SLM
                # predicted_cn8 : code prédit normalisé (on a enlevé les points et espaces)
                # ground_truth_cn8 : véritable code douanier
                # correct : prédiction exacte ou non (sous forme de booléen)
                # reward : récompense obtenue
                # code_status : validité du code prédit

    metrics = {"accuracy": 0, "uncertain": 0, "mean_reward" : 0, "code_status_counts" : {}, "invalid_rate" : 0.0}
    valid_codes = get_valid_codes()
    
    for line in open(dataset_path):
        product = json.loads(line)
        product_description = product["product_description"]
        result = classify_product(product_description)

        code_brut = result["cn_code"]

        predicted_code, code_status = validate_cn8(
            predicted_code=code_brut,
            valid_codes=valid_codes
        )

        reward = calculate_reward(
            predicted_code=predicted_code,
            true_code=product["cn_code"],
            valid_codes=valid_codes
        )

        results["id"].append(product["id"])
        results["product_description"].append(product_description)
        results["status"].append(result["status"])
        results["predicted_cn8"].append(predicted_code)
        results["ground_truth_cn8"].append(product["cn_code"])
        results["correct"].append(code_status == "valid" and predicted_code == product["cn_code"])
        results["raw_predicted"].append(code_brut)
        results["code_status"].append(code_status)
        results["reward"].append(reward)

    metrics["accuracy"] = sum(results["correct"])/len(results["correct"])
    metrics["mean_reward"] = sum(results["reward"]) / len(results["reward"])
    metrics["code_status_counts"] = dict(
        Counter(results["code_status"])
    )

    metrics["invalid_rate"] = (sum(status != "valid"for status in results["code_status"]) / len(results["code_status"])
    )
    return results, metrics

if __name__ == "__main__":
    results, metrics = evaluate()
    print(f"Number of tests: {len(results['product_description'])}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean reward: {metrics['mean_reward']:.4f}")
    print(f"Invalid rate: {metrics['invalid_rate']:.2%}")
