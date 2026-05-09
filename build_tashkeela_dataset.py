#!/usr/bin/env python3
"""Build seq2seq-ready diacritization data from the Tashkeela corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ARABIC_DIACRITIC_CODEPOINTS = {
    0x0610,
    0x0611,
    0x0612,
    0x0613,
    0x0614,
    0x0615,
    0x0616,
    0x0617,
    0x0618,
    0x0619,
    0x061A,
    0x064B,
    0x064C,
    0x064D,
    0x064E,
    0x064F,
    0x0650,
    0x0651,
    0x0652,
    0x0653,
    0x0654,
    0x0655,
    0x0656,
    0x0657,
    0x0658,
    0x0659,
    0x065A,
    0x065B,
    0x065C,
    0x065D,
    0x065E,
    0x065F,
    0x0670,
    0x06D6,
    0x06D7,
    0x06D8,
    0x06D9,
    0x06DA,
    0x06DB,
    0x06DC,
    0x06DF,
    0x06E0,
    0x06E1,
    0x06E2,
    0x06E3,
    0x06E4,
    0x06E7,
    0x06E8,
    0x06EA,
    0x06EB,
    0x06EC,
    0x06ED,
    0x08D3,
    0x08D4,
    0x08D5,
    0x08D6,
    0x08D7,
    0x08D8,
    0x08D9,
    0x08DA,
    0x08DB,
    0x08DC,
    0x08DD,
    0x08DE,
    0x08DF,
    0x08E0,
    0x08E1,
    0x08E3,
    0x08E4,
    0x08E5,
    0x08E6,
    0x08E7,
    0x08E8,
    0x08E9,
    0x08EA,
    0x08EB,
    0x08EC,
    0x08ED,
    0x08EE,
    0x08EF,
    0x08F0,
    0x08F1,
    0x08F2,
    0x08F3,
    0x08F4,
    0x08F5,
    0x08F6,
    0x08F7,
    0x08F8,
    0x08F9,
    0x08FA,
    0x08FB,
    0x08FC,
    0x08FD,
    0x08FE,
    0x08FF,
}
ARABIC_DIACRITICS = frozenset(chr(codepoint) for codepoint in ARABIC_DIACRITIC_CODEPOINTS)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?؟؛:])\s+")
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINE_RE = re.compile(r"\n\s*\n+", re.MULTILINE)
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
ARABIC_WORD_RE = re.compile(
    r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3\u06FA-\u06FC"
    r"\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]+"
)
HEADER_KEYWORDS = (
    "هذا الملف آليا بواسطة",
    "الكتاب :",
    "مؤلف",
    "المؤلف",
    "مصدر الكتاب",
    "http://",
    "https://",
    "الناشر",
    "الطبعة",
    "عدد الأجزاء",
    "ترقيم",
    "الصفحة",
)


@dataclass(frozen=True)
class Document:
    path: Path
    relative_path: str
    domain: str


@dataclass(frozen=True)
class BuilderConfig:
    input_dir: Path
    output_dir: Path
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    min_chars: int
    max_chars: int
    seed: int
    rejected_path: str | None


def parse_args() -> BuilderConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rejected-samples-file",
        default="rejected_samples.jsonl",
        help="Filename for rejected examples inside the output directory. Use empty string to disable.",
    )
    args = parser.parse_args()
    validate_ratios(args.train_ratio, args.validation_ratio, args.test_ratio)
    rejected_path = args.rejected_samples_file or None
    return BuilderConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        seed=args.seed,
        rejected_path=rejected_path,
    )


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    total = train_ratio + validation_ratio + test_ratio
    if any(ratio <= 0 for ratio in (train_ratio, validation_ratio, test_ratio)):
        raise ValueError("All split ratios must be greater than zero.")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0.")


def scan_documents(input_dir: Path) -> list[Document]:
    documents: list[Document] = []
    root = input_dir.resolve()
    for path in sorted(root.rglob("*.txt")):
        relative_path = path.resolve().relative_to(root).as_posix()
        first_part = Path(relative_path).parts[0] if Path(relative_path).parts else ""
        domain = "msa" if first_part.casefold() == "msa" else "classical"
        documents.append(Document(path=path, relative_path=relative_path, domain=domain))
    return documents


def read_document(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def segment_sentences(text: str) -> list[tuple[int, str]]:
    fragments: list[tuple[int, str]] = []
    offset = 0
    for block in BLANK_LINE_RE.split(text):
        block = block.strip()
        if not block:
            offset += 1
            continue
        for sentence in SENTENCE_BOUNDARY_RE.split(block):
            cleaned = sentence.strip()
            if cleaned:
                fragments.append((offset, cleaned))
            offset += 1
    return fragments


def strip_diacritics(text: str) -> str:
    return "".join(char for char in text if char not in ARABIC_DIACRITICS)


def contains_diacritics(text: str) -> bool:
    return any(char in ARABIC_DIACRITICS for char in text)


def arabic_character_count(text: str) -> int:
    return len(ARABIC_CHAR_RE.findall(text))


def arabic_words(text: str) -> list[str]:
    return ARABIC_WORD_RE.findall(text)


def all_arabic_words_have_diacritics(text: str) -> bool:
    words = arabic_words(text)
    if not words:
        return False
    return all(contains_diacritics(word) for word in words)


def likely_header_noise(text: str, position_hint: int) -> bool:
    if position_hint > 12:
        return False
    compact = " ".join(text.split()).casefold()
    if compact.startswith("[") and compact.endswith("]"):
        return True
    return any(keyword in compact for keyword in HEADER_KEYWORDS)


def is_valid_pair(
    source: str,
    target: str,
    *,
    position_hint: int,
    min_chars: int,
    max_chars: int,
) -> tuple[bool, str | None]:
    source_length = len(source)
    if source == target:
        return False, "no_diacritics_removed"
    if source_length < min_chars:
        return False, "too_short"
    if source_length > max_chars:
        return False, "too_long"
    arabic_chars = arabic_character_count(target)
    if arabic_chars == 0:
        return False, "no_arabic_content"
    if arabic_chars / max(len(target), 1) < 0.25:
        return False, "low_arabic_ratio"
    if not contains_diacritics(target):
        return False, "missing_target_diacritics"
    if not all_arabic_words_have_diacritics(target):
        return False, "partial_diacritization"
    if likely_header_noise(target, position_hint):
        return False, "header_noise"
    return True, None


def stable_example_id(relative_path: str, target: str) -> str:
    digest = hashlib.sha256(f"{relative_path}\n{target}".encode("utf-8")).hexdigest()
    return digest[:16]


def build_examples(
    document: Document,
    config: BuilderConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized = normalize_text(read_document(document.path))
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen_targets: set[str] = set()
    for position_hint, target in segment_sentences(normalized):
        source = strip_diacritics(target)
        is_valid, reason = is_valid_pair(
            source,
            target,
            position_hint=position_hint,
            min_chars=config.min_chars,
            max_chars=config.max_chars,
        )
        if not is_valid:
            rejected.append(
                {
                    "domain": document.domain,
                    "source_file": document.relative_path,
                    "reason": reason,
                    "target": target,
                }
            )
            continue
        dedupe_key = f"{document.relative_path}\n{target}"
        if dedupe_key in seen_targets:
            rejected.append(
                {
                    "domain": document.domain,
                    "source_file": document.relative_path,
                    "reason": "duplicate_sentence",
                    "target": target,
                }
            )
            continue
        seen_targets.add(dedupe_key)
        accepted.append(
            {
                "id": stable_example_id(document.relative_path, target),
                "domain": document.domain,
                "source_file": document.relative_path,
                "source": source,
                "target": target,
                "char_length_source": len(source),
                "char_length_target": len(target),
            }
        )
    return accepted, rejected


def split_documents(documents: list[Document], config: BuilderConfig) -> dict[str, list[Document]]:
    shuffled = documents[:]
    random.Random(config.seed).shuffle(shuffled)
    total = len(shuffled)
    if total < 3:
        raise ValueError("At least 3 documents are required to create non-empty train/validation/test splits.")

    train_count = max(1, int(total * config.train_ratio))
    validation_count = max(1, int(total * config.validation_ratio))
    test_count = total - train_count - validation_count

    if test_count < 1:
        overflow = 1 - test_count
        reducible_train = max(0, train_count - 1)
        reduce_from_train = min(overflow, reducible_train)
        train_count -= reduce_from_train
        overflow -= reduce_from_train
        if overflow > 0:
            reducible_validation = max(0, validation_count - 1)
            reduce_from_validation = min(overflow, reducible_validation)
            validation_count -= reduce_from_validation
            overflow -= reduce_from_validation
        test_count = total - train_count - validation_count

    train_cutoff = train_count
    validation_cutoff = train_cutoff + validation_count
    splits = {
        "train": shuffled[:train_cutoff],
        "validation": shuffled[train_cutoff:validation_cutoff],
        "test": shuffled[validation_cutoff:],
    }
    for split_name in ("train", "validation", "test"):
        if not splits[split_name]:
            raise ValueError(f"Split '{split_name}' is empty; adjust ratios or input data size.")
    return splits


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_manifest(
    examples_by_split: dict[str, list[dict[str, object]]],
    split_documents_map: dict[str, list[Document]],
    rejected_count: int,
    config: BuilderConfig,
) -> dict[str, object]:
    split_counts = {split: len(records) for split, records in examples_by_split.items()}
    domain_counts: dict[str, dict[str, int]] = {}
    for split, records in examples_by_split.items():
        counter = Counter(str(record["domain"]) for record in records)
        domain_counts[split] = dict(sorted(counter.items()))
    return {
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "split_ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "filters": {
            "min_chars": config.min_chars,
            "max_chars": config.max_chars,
        },
        "documents_per_split": {
            split: len(documents) for split, documents in split_documents_map.items()
        },
        "examples_per_split": split_counts,
        "domain_counts_per_split": domain_counts,
        "rejected_examples": rejected_count,
        "documents": {
            split: [document.relative_path for document in documents]
            for split, documents in split_documents_map.items()
        },
    }


def write_outputs(
    examples_by_split: dict[str, list[dict[str, object]]],
    rejected_records: list[dict[str, object]],
    split_documents_map: dict[str, list[Document]],
    config: BuilderConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in examples_by_split.items():
        output_path = config.output_dir / f"{split}.jsonl"
        records_with_split = []
        for record in records:
            enriched = dict(record)
            enriched["split"] = split
            records_with_split.append(enriched)
        write_jsonl(output_path, records_with_split)
    if config.rejected_path:
        write_jsonl(config.output_dir / config.rejected_path, rejected_records)
    manifest = build_manifest(examples_by_split, split_documents_map, len(rejected_records), config)
    manifest_path = config.output_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(config: BuilderConfig) -> dict[str, object]:
    documents = scan_documents(config.input_dir)
    if not documents:
        raise ValueError(f"No .txt files were found under {config.input_dir}")
    split_documents_map = split_documents(documents, config)
    examples_by_split: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    rejected_records: list[dict[str, object]] = []
    for split, split_docs in split_documents_map.items():
        for document in split_docs:
            accepted, rejected = build_examples(document, config)
            examples_by_split[split].extend(accepted)
            rejected_records.extend(rejected)
    write_outputs(examples_by_split, rejected_records, split_documents_map, config)
    return build_manifest(examples_by_split, split_documents_map, len(rejected_records), config)


def main() -> None:
    config = parse_args()
    manifest = run(config)
    print(
        json.dumps(
            {
                "documents_per_split": manifest["documents_per_split"],
                "examples_per_split": manifest["examples_per_split"],
                "rejected_examples": manifest["rejected_examples"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
