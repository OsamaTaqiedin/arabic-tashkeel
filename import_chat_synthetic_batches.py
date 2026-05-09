#!/usr/bin/env python3
"""Validate and merge chat-generated synthetic diacritization batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from build_tashkeela_dataset import all_arabic_words_have_diacritics, strip_diacritics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing chat-generated JSON batch files.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Merged accepted synthetic dataset JSONL.",
    )
    parser.add_argument(
        "--rejected-path",
        required=True,
        type=Path,
        help="Rejected records JSONL.",
    )
    parser.add_argument(
        "--source-tag",
        default="chat_manual",
        help="Metadata tag describing the generation source.",
    )
    return parser.parse_args()


def iter_batch_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.json")):
        if path.is_file():
            yield path


def load_batch(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    normalized: list[dict[str, object]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {index} is not a JSON object.")
        normalized.append(item)
    return normalized


def validate_item(item: dict[str, object]) -> tuple[bool, str | None]:
    source = str(item.get("source", ""))
    target = str(item.get("target", ""))
    if not source or not target:
        return False, "missing_source_or_target"
    if strip_diacritics(target) != source:
        return False, "source_target_mismatch"
    if not all_arabic_words_have_diacritics(target):
        return False, "partial_diacritization"
    return True, None


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def merge_batches(input_dir: Path, source_tag: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    for batch_path in iter_batch_files(input_dir):
        for index, item in enumerate(load_batch(batch_path), start=1):
            item_id = str(item.get("id", f"{batch_path.stem}-{index}"))
            source = str(item.get("source", ""))
            target = str(item.get("target", ""))
            pair_key = (source, target)

            is_valid, reason = validate_item(item)
            if not is_valid:
                rejected.append(
                    {
                        "id": item_id,
                        "batch_file": batch_path.name,
                        "source": source,
                        "target": target,
                        "reason": reason,
                    }
                )
                continue

            if item_id in seen_ids or pair_key in seen_pairs:
                rejected.append(
                    {
                        "id": item_id,
                        "batch_file": batch_path.name,
                        "source": source,
                        "target": target,
                        "reason": "duplicate_record",
                    }
                )
                continue

            seen_ids.add(item_id)
            seen_pairs.add(pair_key)
            accepted.append(
                {
                    "id": item_id,
                    "source": source,
                    "target": target,
                    "generator": "chatgpt",
                    "generator_mode": source_tag,
                    "batch_file": batch_path.name,
                }
            )

    return accepted, rejected


def main() -> None:
    args = parse_args()
    accepted, rejected = merge_batches(args.input_dir, args.source_tag)
    write_jsonl(args.output_path, accepted)
    write_jsonl(args.rejected_path, rejected)
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "rejected": len(rejected),
                "input_dir": str(args.input_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
