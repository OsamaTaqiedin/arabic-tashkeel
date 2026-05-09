from __future__ import annotations

from collections import defaultdict


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def diacritic_cer(predictions: list[str], targets: list[str]) -> float:
    total_distance = 0
    total_length = 0
    for prediction, target in zip(predictions, targets, strict=True):
        total_distance += levenshtein_distance(prediction, target)
        total_length += max(len(target), 1)
    return total_distance / max(total_length, 1)


def exact_match_rate(predictions: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    correct = sum(prediction == target for prediction, target in zip(predictions, targets, strict=True))
    return correct / len(targets)


def compute_grouped_metrics(
    predictions: list[str],
    targets: list[str],
    domains: list[str],
) -> dict[str, dict[str, float]]:
    grouped_predictions: dict[str, list[str]] = defaultdict(list)
    grouped_targets: dict[str, list[str]] = defaultdict(list)

    for prediction, target, domain in zip(predictions, targets, domains, strict=True):
        grouped_predictions["overall"].append(prediction)
        grouped_targets["overall"].append(target)
        grouped_predictions[domain].append(prediction)
        grouped_targets[domain].append(target)

    metrics: dict[str, dict[str, float]] = {}
    for group_name, group_targets in grouped_targets.items():
        group_predictions = grouped_predictions[group_name]
        metrics[group_name] = {
            "diacritic_cer": diacritic_cer(group_predictions, group_targets),
            "exact_match": exact_match_rate(group_predictions, group_targets),
            "count": float(len(group_targets)),
        }
    return metrics

