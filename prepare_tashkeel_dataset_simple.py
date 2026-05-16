#!/usr/bin/env python3
"""Minimal Tashkeela dataset preparation script."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ARABIC_DIACRITIC_CODEPOINTS = {
    0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615, 0x0616, 0x0617, 0x0618, 0x0619, 0x061A,
    0x064B, 0x064C, 0x064D, 0x064E, 0x064F, 0x0650, 0x0651, 0x0652, 0x0653, 0x0654, 0x0655,
    0x0656, 0x0657, 0x0658, 0x0659, 0x065A, 0x065B, 0x065C, 0x065D, 0x065E, 0x065F, 0x0670,
    0x06D6, 0x06D7, 0x06D8, 0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2,
    0x06E3, 0x06E4, 0x06E7, 0x06E8, 0x06EA, 0x06EB, 0x06EC, 0x06ED,
    0x08D3, 0x08D4, 0x08D5, 0x08D6, 0x08D7, 0x08D8, 0x08D9, 0x08DA, 0x08DB, 0x08DC, 0x08DD,
    0x08DE, 0x08DF, 0x08E0, 0x08E1, 0x08E3, 0x08E4, 0x08E5, 0x08E6, 0x08E7, 0x08E8, 0x08E9,
    0x08EA, 0x08EB, 0x08EC, 0x08ED, 0x08EE, 0x08EF, 0x08F0, 0x08F1, 0x08F2, 0x08F3, 0x08F4,
    0x08F5, 0x08F6, 0x08F7, 0x08F8, 0x08F9, 0x08FA, 0x08FB, 0x08FC, 0x08FD, 0x08FE, 0x08FF,
}
ARABIC_DIACRITICS = frozenset(chr(codepoint) for codepoint in ARABIC_DIACRITIC_CODEPOINTS)
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINE_RE = re.compile(r"\n\s*\n+", re.MULTILINE)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?؟؛])\s+")
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
ARABIC_WORD_RE = re.compile(
    r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3\u06FA-\u06FC"
    r"\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF]+"
)
HEADER_KEYWORDS = (
    "الكتاب :",
    "المؤلف",
    "مؤلف",
    "الناشر",
    "الطبعة",
    "عدد الأجزاء",
    "مصدر الكتاب",
    "ترقيم",
    "الصفحة",
    "هذا الملف آليا بواسطة",
    "http://",
    "https://",
)


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


DEFAULT_CONFIG = BuilderConfig(
    input_dir=Path(r"C:\Users\User\Downloads\Tashkeela-arabic-diacritized-text-utf8-0.3\texts.txt"),
    output_dir=Path("output_simple"),
    train_ratio=0.9,
    validation_ratio=0.05,
    test_ratio=0.05,
    min_chars=8,
    max_chars=400,
    seed=42,
)


@dataclass(frozen=True)
class Document:
    path: Path
    relative_path: str
    domain: str


def validate_config(config: BuilderConfig) -> None:
    total = config.train_ratio + config.validation_ratio + config.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")
    if config.min_chars < 1:
        raise ValueError("min_chars must be positive.")
    if config.max_chars < config.min_chars:
        raise ValueError("max_chars must be >= min_chars.")


def scan_documents(input_dir: Path) -> list[Document]:
    root = input_dir.resolve()
    documents: list[Document] = []
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
    position = 0
    for block in BLANK_LINE_RE.split(text):
        block = block.strip()
        if not block:
            position += 1
            continue
        for sentence in SENTENCE_BOUNDARY_RE.split(block):
            cleaned = sentence.strip()
            if cleaned:
                fragments.append((position, cleaned))
            position += 1
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
    return bool(words) and all(contains_diacritics(word) for word in words)


def likely_header_noise(text: str, position_hint: int) -> bool:
    if position_hint > 12:
        return False
    compact = " ".join(text.split()).casefold()
    if compact.startswith("[") and compact.endswith("]"):
        return True
    return any(keyword in compact for keyword in HEADER_KEYWORDS)


def is_valid_pair(source: str, target: str, *, position_hint: int, min_chars: int, max_chars: int) -> tuple[bool, str | None]:
    if source == target:
        return False, "no_diacritics_removed"
    if len(source) < min_chars:
        return False, "too_short"
    if len(source) > max_chars:
        return False, "too_long"
    arabic_chars = arabic_character_count(target)
    if arabic_chars == 0:
        return False, "no_arabic_content"
    if arabic_chars / max(len(target), 1) < 0.25:
        return False, "low_arabic_ratio"
    if not all_arabic_words_have_diacritics(target):
        return False, "partial_diacritization"
    if likely_header_noise(target, position_hint):
        return False, "header_noise"
    return True, None


def stable_example_id(relative_path: str, target: str) -> str:
    digest = hashlib.sha256(f"{relative_path}\n{target}".encode("utf-8")).hexdigest()
    return digest[:16]


def build_examples(document: Document, config: BuilderConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    text = normalize_text(read_document(document.path))
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen: set[str] = set()
    for position_hint, target in segment_sentences(text):
        source = strip_diacritics(target)
        valid, reason = is_valid_pair(
            source,
            target,
            position_hint=position_hint,
            min_chars=config.min_chars,
            max_chars=config.max_chars,
        )
        if not valid:
            rejected.append({"source_file": document.relative_path, "domain": document.domain, "reason": reason, "target": target})
            continue
        dedupe_key = f"{document.relative_path}\n{target}"
        if dedupe_key in seen:
            rejected.append({"source_file": document.relative_path, "domain": document.domain, "reason": "duplicate_sentence", "target": target})
            continue
        seen.add(dedupe_key)
        accepted.append(
            {
                "id": stable_example_id(document.relative_path, target),
                "domain": document.domain,
                "source_file": document.relative_path,
                "source": source,
                "target": target,
            }
        )
    return accepted, rejected


def split_documents(documents: list[Document], config: BuilderConfig) -> dict[str, list[Document]]:
    if len(documents) < 3:
        raise ValueError("Need at least 3 documents for train/validation/test splits.")
    shuffled = documents[:]
    random.Random(config.seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = max(1, int(total * config.train_ratio))
    validation_count = max(1, int(total * config.validation_ratio))
    test_count = total - train_count - validation_count
    if test_count < 1:
        if train_count > validation_count:
            train_count -= 1
        else:
            validation_count -= 1
        test_count = total - train_count - validation_count
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_manifest(
    examples_by_split: dict[str, list[dict[str, object]]],
    split_map: dict[str, list[Document]],
    rejected_records: list[dict[str, object]],
    config: BuilderConfig,
) -> dict[str, object]:
    domain_counts: dict[str, dict[str, int]] = {}
    for split, records in examples_by_split.items():
        domain_counts[split] = dict(sorted(Counter(str(record["domain"]) for record in records).items()))
    return {
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "split_ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "filters": {"min_chars": config.min_chars, "max_chars": config.max_chars},
        "documents_per_split": {split: len(documents) for split, documents in split_map.items()},
        "examples_per_split": {split: len(records) for split, records in examples_by_split.items()},
        "domain_counts_per_split": domain_counts,
        "rejected_examples": len(rejected_records),
    }


def run(config: BuilderConfig) -> dict[str, object]:
    validate_config(config)
    documents = scan_documents(config.input_dir)
    if not documents:
        raise ValueError(f"No .txt files found under {config.input_dir}")
    split_map = split_documents(documents, config)
    examples_by_split: dict[str, list[dict[str, object]]] = {"train": [], "validation": [], "test": []}
    rejected_records: list[dict[str, object]] = []
    for split, split_documents_list in split_map.items():
        for document in split_documents_list:
            accepted, rejected = build_examples(document, config)
            for record in accepted:
                record["split"] = split
            examples_by_split[split].extend(accepted)
            rejected_records.extend(rejected)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in examples_by_split.items():
        write_jsonl(config.output_dir / f"{split}.jsonl", records)
    write_jsonl(config.output_dir / "rejected_samples.jsonl", rejected_records)
    manifest = build_manifest(examples_by_split, split_map, rejected_records, config)
    (config.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    manifest = run(DEFAULT_CONFIG)
    print(json.dumps(manifest["examples_per_split"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
