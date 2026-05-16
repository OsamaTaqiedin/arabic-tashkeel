#!/usr/bin/env python3
"""Minimal GRU attention seq2seq trainer for Arabic tashkeel."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
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
ARABIC_BASE_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3\u06FA-\u06FC]")


@dataclass(frozen=True)
class Example:
    source: str
    target: str
    domain: str


@dataclass
class TrainConfig:
    train_path: str
    validation_path: str
    test_path: str | None
    checkpoint_dir: str
    embedding_dim: int
    encoder_hidden_size: int
    decoder_hidden_size: int
    batch_size: int
    learning_rate: float
    epochs: int
    teacher_forcing_ratio: float
    dropout: float
    max_decode_length: int
    device: str
    max_train_examples: int | None
    max_validation_examples: int | None
    max_test_examples: int | None
    run_test: bool


DEFAULT_CONFIG = TrainConfig(
    train_path="output_simple/train.jsonl",
    validation_path="output_simple/validation.jsonl",
    test_path="output_simple/test.jsonl",
    checkpoint_dir="artifacts/gru_attention_simple",
    embedding_dim=64,
    encoder_hidden_size=64,
    decoder_hidden_size=128,
    batch_size=16,
    learning_rate=1e-3,
    epochs=1,
    teacher_forcing_ratio=0.5,
    dropout=0.1,
    max_decode_length=420,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_train_examples=None,
    max_validation_examples=None,
    max_test_examples=None,
    run_test=True,
)


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

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(char, self.unk_id) for char in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        output: list[str] = []
        for token_id in ids:
            token = self.itos.get(int(token_id), UNK_TOKEN)
            if token == EOS_TOKEN:
                break
            if token in SPECIAL_TOKENS:
                continue
            output.append(token)
        return "".join(output)

    def to_dict(self) -> dict[str, object]:
        return {"tokens": self.tokens}


class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, examples: list[Example], vocab: CharVocabulary) -> None:
        self.examples = examples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        return {
            "source_text": example.source,
            "target_text": example.target,
            "domain": example.domain,
            "source_ids": self.vocab.encode(example.source, add_eos=True),
            "target_ids": self.vocab.encode(example.target, add_bos=True, add_eos=True),
        }


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, source_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected_encoder = self.encoder_projection(encoder_outputs)
        projected_decoder = self.decoder_projection(decoder_hidden).unsqueeze(1)
        energy = torch.tanh(projected_encoder + projected_decoder)
        scores = self.score(energy).squeeze(-1)
        scores = scores.masked_fill(~source_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class GRUAttentionSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embedding_dim, encoder_hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(embedding_dim + encoder_hidden_size * 2, decoder_hidden_size, batch_first=True)
        self.attention = AdditiveAttention(encoder_hidden_size * 2, decoder_hidden_size, decoder_hidden_size)
        self.encoder_to_decoder = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
        self.output_projection = nn.Linear(decoder_hidden_size + encoder_hidden_size * 2 + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, source_ids: torch.Tensor, source_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, hidden = self.encoder(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        decoder_hidden = torch.tanh(self.encoder_to_decoder(hidden)).unsqueeze(0)
        return encoder_outputs, decoder_hidden

    def decode_step(
        self,
        previous_tokens: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(previous_tokens)).unsqueeze(1)
        context, _ = self.attention(hidden[-1], encoder_outputs, source_mask)
        decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        decoder_output, hidden = self.decoder(decoder_input, hidden)
        decoder_output = decoder_output.squeeze(1)
        logits = self.output_projection(torch.cat([decoder_output, context, embedded.squeeze(1)], dim=-1))
        return logits, hidden

    def forward(self, source_ids: torch.Tensor, source_lengths: torch.Tensor, target_ids: torch.Tensor, teacher_forcing_ratio: float) -> torch.Tensor:
        encoder_outputs, hidden = self.encode(source_ids, source_lengths)
        source_mask = source_ids.ne(self.pad_id)
        batch_size = source_ids.size(0)
        target_steps = target_ids.size(1) - 1
        vocab_size = self.output_projection.out_features
        logits = torch.zeros(batch_size, target_steps, vocab_size, device=source_ids.device)
        decoder_input = target_ids[:, 0]

        for step in range(target_steps):
            step_logits, hidden = self.decode_step(decoder_input, hidden, encoder_outputs, source_mask)
            logits[:, step, :] = step_logits
            use_teacher = random.random() < teacher_forcing_ratio
            predicted = step_logits.argmax(dim=-1)
            decoder_input = target_ids[:, step + 1] if use_teacher else predicted
        return logits

    @torch.no_grad()
    def greedy_decode(self, source_ids: torch.Tensor, source_lengths: torch.Tensor, max_decode_length: int) -> torch.Tensor:
        encoder_outputs, hidden = self.encode(source_ids, source_lengths)
        source_mask = source_ids.ne(self.pad_id)
        batch_size = source_ids.size(0)
        decoder_input = torch.full((batch_size,), self.bos_id, dtype=torch.long, device=source_ids.device)
        generated: list[torch.Tensor] = []
        for _ in range(max_decode_length):
            step_logits, hidden = self.decode_step(decoder_input, hidden, encoder_outputs, source_mask)
            decoder_input = step_logits.argmax(dim=-1)
            generated.append(decoder_input)
            if torch.all(decoder_input.eq(self.eos_id)):
                break
        if not generated:
            return torch.empty(batch_size, 0, dtype=torch.long, device=source_ids.device)
        return torch.stack(generated, dim=1)


def validate_config(config: TrainConfig) -> None:
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    if config.embedding_dim < 1 or config.encoder_hidden_size < 1 or config.decoder_hidden_size < 1:
        raise ValueError("Model dimensions must be positive.")
    if not config.train_path or not config.validation_path or not config.checkpoint_dir:
        raise ValueError("train_path, validation_path, and checkpoint_dir must be set.")


def load_jsonl(path: str | Path, limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(
                Example(
                    source=str(record["source"]),
                    target=str(record["target"]),
                    domain=str(record.get("domain", "unknown")),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def build_vocab(examples: Iterable[Example]) -> CharVocabulary:
    counts: Counter[str] = Counter()
    for example in examples:
        counts.update(example.source)
        counts.update(example.target)
    tokens = SPECIAL_TOKENS[:] + sorted(counts)
    return CharVocabulary(tokens)


def collate_batch(items: list[dict[str, object]], pad_id: int) -> dict[str, object]:
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
        "source_ids": torch.tensor(source_batch, dtype=torch.long),
        "target_ids": torch.tensor(target_batch, dtype=torch.long),
        "source_lengths": torch.tensor(source_lengths, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "target_texts": [item["target_text"] for item in items],
        "domains": [item["domain"] for item in items],
    }


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
            current.append(min(
                current[right_index - 1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


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
    words: list[list[tuple[str, str]]] = []
    for token in text.split():
        units = _arabic_letter_units(token)
        if units:
            words.append(units)
    return words


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
    return sum(prediction == target for prediction, target in zip(predictions, targets, strict=True)) / len(targets)


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


def compute_grouped_metrics(predictions: list[str], targets: list[str], domains: list[str]) -> dict[str, dict[str, float]]:
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


def evaluate(model: GRUAttentionSeq2Seq, loader: DataLoader, criterion: nn.Module, vocab: CharVocabulary, device: torch.device, max_decode_length: int) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions: list[str] = []
    targets: list[str] = []
    domains: list[str] = []
    with torch.no_grad():
        for batch in loader:
            source_ids = batch["source_ids"].to(device)
            source_lengths = batch["source_lengths"].to(device)
            target_ids = batch["target_ids"].to(device)
            logits = model(source_ids, source_lengths, target_ids, teacher_forcing_ratio=0.0)
            gold = target_ids[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            total_loss += float(loss.item())
            total_batches += 1
            decoded = model.greedy_decode(source_ids, source_lengths, max_decode_length=max_decode_length)
            predictions.extend(vocab.decode(row.tolist()) for row in decoded.cpu())
            targets.extend(batch["target_texts"])
            domains.extend(batch["domains"])
    return {
        "loss": total_loss / max(total_batches, 1),
        "metrics": compute_grouped_metrics(predictions, targets, domains),
    }


def save_vocab(vocab: CharVocabulary, path: Path) -> None:
    path.write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_model(config: TrainConfig) -> dict[str, object]:
    validate_config(config)
    train_examples = load_jsonl(config.train_path, config.max_train_examples)
    validation_examples = load_jsonl(config.validation_path, config.max_validation_examples)
    test_examples = load_jsonl(config.test_path, config.max_test_examples) if config.test_path else []

    vocab = build_vocab(train_examples)
    train_dataset = JsonlSeq2SeqDataset(train_examples, vocab)
    validation_dataset = JsonlSeq2SeqDataset(validation_examples, vocab)
    test_dataset = JsonlSeq2SeqDataset(test_examples, vocab) if test_examples else None

    collate_fn = lambda items: collate_batch(items, vocab.pad_id)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn) if test_dataset else None

    device = torch.device(config.device)
    model = GRUAttentionSeq2Seq(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        encoder_hidden_size=config.encoder_hidden_size,
        decoder_hidden_size=config.decoder_hidden_size,
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, checkpoint_dir / "vocab.json")
    (checkpoint_dir / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history: list[dict[str, object]] = []
    best_validation_der = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            source_ids = batch["source_ids"].to(device)
            source_lengths = batch["source_lengths"].to(device)
            target_ids = batch["target_ids"].to(device)
            logits = model(source_ids, source_lengths, target_ids, teacher_forcing_ratio=config.teacher_forcing_ratio)
            targets = target_ids[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        validation = evaluate(model, validation_loader, criterion, vocab, device, config.max_decode_length)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_batches, 1),
            "validation_loss": validation["loss"],
            "validation_metrics": validation["metrics"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))
        validation_der = validation["metrics"]["overall"]["diacritic_error_rate"]
        if validation_der < best_validation_der:
            best_validation_der = validation_der
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, checkpoint_dir / "best.pt")

    result: dict[str, object] = {"history": history, "best_validation_der": best_validation_der}
    if config.run_test and test_loader is not None:
        best_state = torch.load(checkpoint_dir / "best.pt", map_location=device)
        model.load_state_dict(best_state["model_state_dict"])
        result["test"] = evaluate(model, test_loader, criterion, vocab, device, config.max_decode_length)
        print(json.dumps({"test": result["test"]}, ensure_ascii=False))

    (checkpoint_dir / "history.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    train_model(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
