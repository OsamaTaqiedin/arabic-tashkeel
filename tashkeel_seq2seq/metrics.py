from __future__ import annotations

from collections import defaultdict
import re

from build_tashkeela_dataset import ARABIC_DIACRITICS


ARABIC_BASE_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3\u06FA-\u06FC]")


def _arabic_letter_units(token: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    current_base: str | None = None
    current_diacritics: list[str] = []

    for char in token:
        if char in ARABIC_DIACRITICS:
            if current_base is not None:
                current_diacritics.append(char)
            continue
        if ARABIC_BASE_RE.fullmatch(char):
            if current_base is not None:
                units.append((current_base, "".join(current_diacritics)))
            current_base = char
            current_diacritics = []
            continue
        if current_base is not None:
            units.append((current_base, "".join(current_diacritics)))
            current_base = None
            current_diacritics = []

    if current_base is not None:
        units.append((current_base, "".join(current_diacritics)))
    return units


def _arabic_word_units(text: str) -> list[list[tuple[str, str]]]:
    words = []
    for token in text.split():
        units = _arabic_letter_units(token)
        if units:
            words.append(units)
    return words


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


def diacritic_error_rate(predictions: list[str], targets: list[str]) -> float:
    wrong = 0
    total = 0

    for prediction, target in zip(predictions, targets, strict=True):
        predicted_words = _arabic_word_units(prediction)
        target_words = _arabic_word_units(target)
        for word_index, target_units in enumerate(target_words):
            predicted_units = predicted_words[word_index] if word_index < len(predicted_words) else []
            for unit_index, target_unit in enumerate(target_units):
                total += 1
                if unit_index >= len(predicted_units):
                    wrong += 1
                    continue
                predicted_unit = predicted_units[unit_index]
                if predicted_unit[0] != target_unit[0] or predicted_unit[1] != target_unit[1]:
                    wrong += 1
    return wrong / max(total, 1)


def word_error_rate(predictions: list[str], targets: list[str]) -> float:
    wrong = 0
    total = 0

    for prediction, target in zip(predictions, targets, strict=True):
        predicted_words = _arabic_word_units(prediction)
        target_words = _arabic_word_units(target)
        for word_index, target_units in enumerate(target_words):
            total += 1
            if word_index >= len(predicted_words) or predicted_words[word_index] != target_units:
                wrong += 1
    return wrong / max(total, 1)


def case_ending_error_rate(predictions: list[str], targets: list[str]) -> float:
    wrong = 0
    total = 0

    for prediction, target in zip(predictions, targets, strict=True):
        predicted_words = _arabic_word_units(prediction)
        target_words = _arabic_word_units(target)
        for word_index, target_units in enumerate(target_words):
            if not target_units:
                continue
            total += 1
            if word_index >= len(predicted_words) or not predicted_words[word_index]:
                wrong += 1
                continue
            predicted_unit = predicted_words[word_index][-1]
            target_unit = target_units[-1]
            if predicted_unit[0] != target_unit[0] or predicted_unit[1] != target_unit[1]:
                wrong += 1
    return wrong / max(total, 1)


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
            "diacritic_error_rate": diacritic_error_rate(group_predictions, group_targets),
            "word_error_rate": word_error_rate(group_predictions, group_targets),
            "case_ending_error_rate": case_ending_error_rate(group_predictions, group_targets),
            "exact_match": exact_match_rate(group_predictions, group_targets),
            "count": float(len(group_targets)),
        }
    return metrics
