from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import torch
except ImportError:  # pragma: no cover - handled in runtime/test skips
    torch = None


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


@dataclass(frozen=True)
class TashkeelExample:
    id: str
    source: str
    target: str
    domain: str
    source_file: str
    split: str


class CharVocabulary:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.stoi = {token: index for index, token in enumerate(tokens)}
        self.itos = {index: token for index, token in enumerate(tokens)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(char, self.unk_id) for char in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, stop_at_eos: bool = True, skip_special: bool = True) -> str:
        decoded: list[str] = []
        for token_id in ids:
            token = self.itos.get(int(token_id), UNK_TOKEN)
            if stop_at_eos and token == EOS_TOKEN:
                break
            if skip_special and token in SPECIAL_TOKENS:
                continue
            decoded.append(token)
        return "".join(decoded)

    def to_dict(self) -> dict[str, object]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CharVocabulary":
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens):
            raise ValueError("Invalid vocabulary payload.")
        return cls(list(tokens))


def load_jsonl(path: Path) -> list[TashkeelExample]:
    examples: list[TashkeelExample] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                examples.append(
                    TashkeelExample(
                        id=str(record["id"]),
                        source=str(record["source"]),
                        target=str(record["target"]),
                        domain=str(record["domain"]),
                        source_file=str(record["source_file"]),
                        split=str(record["split"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing field {exc} in {path} line {line_number}") from exc
    return examples


def build_char_vocab(examples: Iterable[TashkeelExample], *, min_frequency: int = 1) -> CharVocabulary:
    counts: Counter[str] = Counter()
    for example in examples:
        counts.update(example.source)
        counts.update(example.target)
    tokens = SPECIAL_TOKENS[:]
    tokens.extend(sorted(char for char, count in counts.items() if count >= min_frequency))
    return CharVocabulary(tokens)


class TashkeelDataset:
    def __init__(self, examples: list[TashkeelExample], vocab: CharVocabulary) -> None:
        self.examples = examples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        source_ids = self.vocab.encode(example.source, add_bos=False, add_eos=True)
        target_ids = self.vocab.encode(example.target, add_bos=True, add_eos=True)
        return {
            "id": example.id,
            "source_text": example.source,
            "target_text": example.target,
            "domain": example.domain,
            "source_file": example.source_file,
            "split": example.split,
            "source_ids": source_ids,
            "target_ids": target_ids,
        }


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for batching and training.")


def collate_batch(items: list[dict[str, object]], pad_id: int) -> dict[str, object]:
    _require_torch()
    items = sorted(items, key=lambda item: len(item["source_ids"]), reverse=True)
    source_lengths = [len(item["source_ids"]) for item in items]
    target_lengths = [len(item["target_ids"]) for item in items]
    max_source = max(source_lengths)
    max_target = max(target_lengths)

    source_batch = []
    target_batch = []
    for item in items:
        source_ids = list(item["source_ids"])
        target_ids = list(item["target_ids"])
        source_batch.append(source_ids + [pad_id] * (max_source - len(source_ids)))
        target_batch.append(target_ids + [pad_id] * (max_target - len(target_ids)))

    return {
        "ids": [item["id"] for item in items],
        "source_texts": [item["source_text"] for item in items],
        "target_texts": [item["target_text"] for item in items],
        "domains": [item["domain"] for item in items],
        "source_files": [item["source_file"] for item in items],
        "source_ids": torch.tensor(source_batch, dtype=torch.long),
        "target_ids": torch.tensor(target_batch, dtype=torch.long),
        "source_lengths": torch.tensor(source_lengths, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
    }


def make_collate_fn(vocab: CharVocabulary):
    def _collate(items: list[dict[str, object]]) -> dict[str, object]:
        return collate_batch(items, vocab.pad_id)

    return _collate


def save_vocab(vocab: CharVocabulary, path: Path) -> None:
    path.write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_vocab(path: Path) -> CharVocabulary:
    return CharVocabulary.from_dict(json.loads(path.read_text(encoding="utf-8")))
