from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import TashkeelDataset, build_char_vocab, load_jsonl, make_collate_fn, save_vocab
from .metrics import compute_grouped_metrics
from .simple_model import SimpleCharSeq2Seq


@dataclass
class SimpleTrainingConfig:
    train_path: str
    validation_path: str
    test_path: str
    checkpoint_dir: str
    embedding_dim: int
    hidden_size: int
    batch_size: int
    learning_rate: float
    epochs: int
    teacher_forcing_ratio: float
    max_decode_length: int
    device: str
    run_test: bool


def parse_args() -> SimpleTrainingConfig:
    parser = argparse.ArgumentParser(description="Train a simple character-level GRU seq2seq model.")
    parser.add_argument("--train-path", default="output_strict/train.jsonl")
    parser.add_argument("--validation-path", default="output_strict/validation.jsonl")
    parser.add_argument("--test-path", default="output_strict/test.jsonl")
    parser.add_argument("--checkpoint-dir", default="artifacts/simple_char_seq2seq")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.7)
    parser.add_argument("--max-decode-length", type=int, default=420)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-test", action="store_true")
    return SimpleTrainingConfig(**vars(parser.parse_args()))


def build_dataloaders(config: SimpleTrainingConfig):
    train_examples = load_jsonl(Path(config.train_path))
    validation_examples = load_jsonl(Path(config.validation_path))
    test_examples = load_jsonl(Path(config.test_path))
    vocab = build_char_vocab(train_examples)
    collate_fn = make_collate_fn(vocab)
    train_loader = DataLoader(TashkeelDataset(train_examples, vocab), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(TashkeelDataset(validation_examples, vocab), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(TashkeelDataset(test_examples, vocab), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    return vocab, train_loader, validation_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion, vocab, device, max_decode_length):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions: list[str] = []
    targets: list[str] = []
    domains: list[str] = []

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


def train_model(config: SimpleTrainingConfig) -> dict[str, object]:
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab, train_loader, validation_loader, test_loader = build_dataloaders(config)
    save_vocab(vocab, checkpoint_dir / "vocab.json")
    (checkpoint_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    device = torch.device(config.device)
    model = SimpleCharSeq2Seq(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    history = []
    best_validation_cer = float("inf")

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
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, checkpoint_dir / "latest.pt")
        validation_cer = validation["metrics"]["overall"]["diacritic_cer"]
        if validation_cer < best_validation_cer:
            best_validation_cer = validation_cer
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, checkpoint_dir / "best.pt")

    result = {"history": history, "best_validation_cer": best_validation_cer}
    if config.run_test:
        best_state = torch.load(checkpoint_dir / "best.pt", map_location=device)
        model.load_state_dict(best_state["model_state_dict"])
        result["test"] = evaluate(model, test_loader, criterion, vocab, device, config.max_decode_length)
        print(json.dumps({"test": result["test"]}, ensure_ascii=False))

    (checkpoint_dir / "history.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    train_model(parse_args())


if __name__ == "__main__":
    main()
