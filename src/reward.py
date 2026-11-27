
import re


def calculate_reward(
    predicted_code: str | None,
    true_code: str,
    valid_codes: set[str] | None = None,
) -> float:
    """Reward the longest correct CN prefix; validity is measured separately."""
    if predicted_code is None:
        return 0.0
    if not re.fullmatch(r"\d{8}", predicted_code):
        return 0.0

    if predicted_code == true_code:
        return 1.0

    rewards = {
        6: 0.6,
        4: 0.4,
        2: 0.2,
    }

    for depth, reward in rewards.items():
        if predicted_code[:depth] == true_code[:depth]:
            return reward

    return 0.0
