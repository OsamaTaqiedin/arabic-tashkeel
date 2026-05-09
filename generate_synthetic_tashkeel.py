#!/usr/bin/env python3
"""Generate synthetic Arabic diacritization data using the OpenAI API."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from build_tashkeela_dataset import all_arabic_words_have_diacritics, strip_diacritics


PROMPT_VERSION = "v1"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_SYSTEM_PROMPT = (
    "You generate fully diacritized Modern Standard Arabic text. "
    "Given undiacritized Arabic input, return JSON only with fields "
    "\"source\" and \"target\". "
    "Rules: preserve the wording, do not summarize, do not explain, do not add content, "
    "and ensure every Arabic word in target is fully diacritized."
)


@dataclass(frozen=True)
class SyntheticConfig:
    input_path: Path
    output_path: Path
    rejected_path: Path | None
    model: str
    prompt: str
    max_items: int | None
    sleep_seconds: float
    api_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-batch", help="Prepare OpenAI Batch API requests JSONL.")
    add_common_input_arguments(prepare)
    prepare.add_argument("--output-path", required=True, type=Path)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--max-items", type=int, default=None)
    prepare.add_argument("--prompt", default=DEFAULT_SYSTEM_PROMPT)

    merge = subparsers.add_parser("merge-batch", help="Merge OpenAI Batch API results into dataset JSONL.")
    merge.add_argument("--batch-output-path", required=True, type=Path)
    merge.add_argument("--output-path", required=True, type=Path)
    merge.add_argument("--rejected-path", type=Path, default=None)
    merge.add_argument("--source-tag", default="openai_batch")
    merge.add_argument("--model", default=DEFAULT_MODEL)

    synthesize = subparsers.add_parser("synthesize", help="Call the OpenAI API directly for a smaller run.")
    add_common_input_arguments(synthesize)
    synthesize.add_argument("--output-path", required=True, type=Path)
    synthesize.add_argument("--rejected-path", type=Path, default=None)
    synthesize.add_argument("--model", default=DEFAULT_MODEL)
    synthesize.add_argument("--max-items", type=int, default=None)
    synthesize.add_argument("--sleep-seconds", type=float, default=0.0)
    synthesize.add_argument("--prompt", default=DEFAULT_SYSTEM_PROMPT)
    synthesize.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))

    return parser.parse_args()


def add_common_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-path",
        required=True,
        type=Path,
        help="Input JSONL with at least a 'source' field containing undiacritized Arabic text.",
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


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def make_request_payload(source_text: str, model: str, prompt: str) -> dict[str, object]:
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Diacritize this Arabic text and return JSON only.\n"
                            f"source: {source_text}"
                        ),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "synthetic_tashkeel_pair",
                "schema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["source", "target"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        },
    }


def record_to_batch_request(
    record: dict[str, object],
    index: int,
    *,
    model: str,
    prompt: str,
) -> dict[str, object]:
    source_text = str(record["source"])
    return {
        "custom_id": str(record.get("id", f"synthetic-{index}")),
        "method": "POST",
        "url": "/v1/responses",
        "body": make_request_payload(source_text, model, prompt),
    }


def extract_source_records(input_path: Path, max_items: int | None) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for index, record in enumerate(iter_jsonl(input_path), start=1):
        if "source" not in record:
            raise ValueError(f"Record {index} in {input_path} is missing 'source'.")
        items.append(record)
        if max_items is not None and len(items) >= max_items:
            break
    return items


def prepare_batch(input_path: Path, output_path: Path, model: str, prompt: str, max_items: int | None) -> int:
    source_records = extract_source_records(input_path, max_items)
    requests = (
        record_to_batch_request(record, index, model=model, prompt=prompt)
        for index, record in enumerate(source_records, start=1)
    )
    return write_jsonl(output_path, requests)


def parse_response_output(response_body: dict[str, object]) -> dict[str, str]:
    text_section = response_body.get("output_text")
    if isinstance(text_section, str):
        return json.loads(text_section)

    output = response_body.get("output", [])
    if not isinstance(output, list):
        raise ValueError("Unexpected response payload: missing output content.")

    collected_text: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                collected_text.append(content["text"])
    if not collected_text:
        raise ValueError("No JSON text found in response payload.")
    return json.loads("".join(collected_text))


def validate_synthetic_pair(source_text: str, target_text: str) -> tuple[bool, str | None]:
    if strip_diacritics(target_text) != source_text:
        return False, "source_target_mismatch"
    if not all_arabic_words_have_diacritics(target_text):
        return False, "partial_diacritization"
    return True, None


def openai_post(payload: dict[str, object], api_key: str) -> dict[str, object]:
    request = urllib.request.Request(
        url="https://api.openai.com/v1/responses",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload).encode("utf-8"),
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def synthesize(config: SyntheticConfig) -> tuple[int, int]:
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY is required for the synthesize command.")

    accepted_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    source_records = extract_source_records(config.input_path, config.max_items)

    for index, record in enumerate(source_records, start=1):
        source_text = str(record["source"])
        payload = make_request_payload(source_text, config.model, config.prompt)
        try:
            response = openai_post(payload, config.api_key)
            parsed = parse_response_output(response)
            target_text = str(parsed["target"])
            is_valid, reason = validate_synthetic_pair(source_text, target_text)
            if is_valid:
                accepted_records.append(
                    {
                        "id": str(record.get("id", f"synthetic-{index}")),
                        "source": source_text,
                        "target": target_text,
                        "generator": "openai",
                        "generator_mode": "responses_sync",
                        "generator_model": config.model,
                        "prompt_version": PROMPT_VERSION,
                        "source_origin": record.get("source_file"),
                    }
                )
            else:
                rejected_records.append(
                    {
                        "id": str(record.get("id", f"synthetic-{index}")),
                        "source": source_text,
                        "target": target_text,
                        "reason": reason,
                        "generator_model": config.model,
                    }
                )
        except (KeyError, ValueError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            rejected_records.append(
                {
                    "id": str(record.get("id", f"synthetic-{index}")),
                    "source": source_text,
                    "reason": "api_error",
                    "error": str(exc),
                    "generator_model": config.model,
                }
            )
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    write_jsonl(config.output_path, accepted_records)
    if config.rejected_path is not None:
        write_jsonl(config.rejected_path, rejected_records)
    return len(accepted_records), len(rejected_records)


def merge_batch(batch_output_path: Path, output_path: Path, rejected_path: Path | None, source_tag: str, model: str) -> tuple[int, int]:
    accepted_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []

    for record in iter_jsonl(batch_output_path):
        custom_id = str(record.get("custom_id", ""))
        response_body = record.get("response", {}).get("body")
        if not isinstance(response_body, dict):
            rejected_records.append({"id": custom_id, "reason": "missing_response_body"})
            continue
        try:
            parsed = parse_response_output(response_body)
            source_text = str(parsed["source"])
            target_text = str(parsed["target"])
            is_valid, reason = validate_synthetic_pair(source_text, target_text)
            if is_valid:
                accepted_records.append(
                    {
                        "id": custom_id,
                        "source": source_text,
                        "target": target_text,
                        "generator": "openai",
                        "generator_mode": source_tag,
                        "generator_model": model,
                        "prompt_version": PROMPT_VERSION,
                    }
                )
            else:
                rejected_records.append(
                    {
                        "id": custom_id,
                        "source": source_text,
                        "target": target_text,
                        "reason": reason,
                        "generator_model": model,
                    }
                )
        except (KeyError, ValueError, TypeError) as exc:
            rejected_records.append({"id": custom_id, "reason": "parse_error", "error": str(exc)})

    write_jsonl(output_path, accepted_records)
    if rejected_path is not None:
        write_jsonl(rejected_path, rejected_records)
    return len(accepted_records), len(rejected_records)


def main() -> None:
    args = parse_args()
    if args.command == "prepare-batch":
        count = prepare_batch(args.input_path, args.output_path, args.model, args.prompt, args.max_items)
        print(json.dumps({"prepared_requests": count}, ensure_ascii=False, indent=2))
        return

    if args.command == "merge-batch":
        accepted, rejected = merge_batch(
            args.batch_output_path,
            args.output_path,
            args.rejected_path,
            args.source_tag,
            args.model,
        )
        print(json.dumps({"accepted": accepted, "rejected": rejected}, ensure_ascii=False, indent=2))
        return

    config = SyntheticConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        rejected_path=args.rejected_path,
        model=args.model,
        prompt=args.prompt,
        max_items=args.max_items,
        sleep_seconds=args.sleep_seconds,
        api_key=args.api_key,
    )
    accepted, rejected = synthesize(config)
    print(json.dumps({"accepted": accepted, "rejected": rejected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
