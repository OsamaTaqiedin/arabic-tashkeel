#!/usr/bin/env python3
"""Generate synthetic diacritization pairs by batching calls through Codex CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from build_tashkeela_dataset import all_arabic_words_have_diacritics, strip_diacritics


DEFAULT_MODEL = "gpt-5.4-mini"
PROMPT_VERSION = "codex_cli_v1"


@dataclass(frozen=True)
class CliConfig:
    input_path: Path
    output_path: Path
    rejected_path: Path
    model: str
    batch_size: int
    max_items: int | None
    max_batches: int | None
    codex_bin: str


def parse_args() -> CliConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--rejected-path", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--codex-bin", default="codex")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    return CliConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        rejected_path=args.rejected_path,
        model=args.model,
        batch_size=args.batch_size,
        max_items=args.max_items,
        max_batches=args.max_batches,
        codex_bin=args.codex_bin,
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc


def load_source_records(path: Path, max_items: int | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, record in enumerate(iter_jsonl(path), start=1):
        if "source" not in record:
            raise ValueError(f"Record {index} in {path} is missing 'source'.")
        records.append(record)
        if max_items is not None and len(records) >= max_items:
            break
    return records


def chunk_records(records: list[dict[str, object]], batch_size: int, max_batches: int | None) -> list[list[dict[str, object]]]:
    batches = [records[index : index + batch_size] for index in range(0, len(records), batch_size)]
    if max_batches is not None:
        return batches[:max_batches]
    return batches


def build_prompt(batch: list[dict[str, object]]) -> str:
    lines = [
        "Generate fully diacritized Modern Standard Arabic targets for each source.",
        "Return only a JSON array.",
        "Each item must contain exactly: id, source, target.",
        "Preserve the source wording exactly in target after removing diacritics.",
        "Every Arabic word in target must be diacritized.",
        "",
        "Items:",
    ]
    for record in batch:
        lines.append(f"- id: {record['id']}")
        lines.append(f"  source: {record['source']}")
    return "\n".join(lines)


def schema_payload() -> dict[str, object]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "source": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["id", "source", "target"],
            "additionalProperties": False,
        },
    }


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


def run_codex_batch(codex_bin: str, model: str, prompt: str) -> list[dict[str, object]]:
    codex_path = shutil.which(codex_bin)
    if codex_path is None:
        raise FileNotFoundError(f"Could not find Codex CLI binary '{codex_bin}' in PATH.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        schema_path = temp_root / "schema.json"
        output_path = temp_root / "response.json"
        schema_path.write_text(json.dumps(schema_payload(), ensure_ascii=False), encoding="utf-8")
        command = [
            codex_path,
            "exec",
            "--model",
            model,
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            prompt,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Codex CLI batch failed.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return json.loads(output_path.read_text(encoding="utf-8"))


def generate(config: CliConfig) -> tuple[int, int]:
    source_records = load_source_records(config.input_path, config.max_items)
    for index, record in enumerate(source_records, start=1):
        record.setdefault("id", f"cli-{index:08d}")

    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for batch_index, batch in enumerate(
        chunk_records(source_records, config.batch_size, config.max_batches),
        start=1,
    ):
        prompt = build_prompt(batch)
        try:
            generated = run_codex_batch(config.codex_bin, config.model, prompt)
        except Exception as exc:
            for record in batch:
                rejected.append(
                    {
                        "id": str(record["id"]),
                        "source": str(record["source"]),
                        "reason": "codex_cli_error",
                        "error": str(exc),
                        "batch_index": batch_index,
                    }
                )
            continue

        for item in generated:
            item_id = str(item.get("id", ""))
            is_valid, reason = validate_item(item)
            if not item_id:
                rejected.append(
                    {
                        "id": "",
                        "source": str(item.get("source", "")),
                        "target": str(item.get("target", "")),
                        "reason": "missing_id",
                        "batch_index": batch_index,
                    }
                )
                continue
            if item_id in seen_ids:
                rejected.append(
                    {
                        "id": item_id,
                        "source": str(item.get("source", "")),
                        "target": str(item.get("target", "")),
                        "reason": "duplicate_id",
                        "batch_index": batch_index,
                    }
                )
                continue
            if not is_valid:
                rejected.append(
                    {
                        "id": item_id,
                        "source": str(item.get("source", "")),
                        "target": str(item.get("target", "")),
                        "reason": reason,
                        "batch_index": batch_index,
                    }
                )
                continue
            seen_ids.add(item_id)
            accepted.append(
                {
                    "id": item_id,
                    "source": str(item["source"]),
                    "target": str(item["target"]),
                    "generator": "codex_cli",
                    "generator_model": config.model,
                    "prompt_version": PROMPT_VERSION,
                    "batch_index": batch_index,
                }
            )

    write_jsonl(config.output_path, accepted)
    write_jsonl(config.rejected_path, rejected)
    return len(accepted), len(rejected)


def main() -> None:
    config = parse_args()
    accepted, rejected = generate(config)
    print(json.dumps({"accepted": accepted, "rejected": rejected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
