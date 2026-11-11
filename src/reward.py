
import re


def calculate_reward(predicted_code: str | None,true_code: str,valid_codes: set[str] | None = None) -> float:
    """Calcule une récompense selon la hiérarchie douanière, on donne une récompense proportionnelle à son niveau de réponse dans la hiérarchie."""
    if predicted_code is None:
        return 0.0
    if not re.fullmatch(r"\d{8}", predicted_code):
        return 0.0

    if valid_codes is not None and predicted_code not in valid_codes:
        return 0.0

    if predicted_code == true_code:
        return 1.0

    rewards = {
        6: 0.6, 
        4: 0.3, 
        2: 0.1, 
    }

    for depth, reward in rewards.items():
        if predicted_code[:depth] == true_code[:depth]:
            return reward

    return 0.0
