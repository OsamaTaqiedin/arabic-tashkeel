from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import TashkeelDataset, build_char_vocab, load_jsonl, make_collate_fn, save_vocab
from .metrics import compute_grouped_metrics
from .model import GRUSeq2Seq


@dataclass
class TrainingConfig:
    train_path: str
    validation_path: str
    test_path: str
    checkpoint_dir: str
    embedding_dim: int
    encoder_hidden_size: int
    decoder_hidden_size: int
    batch_size: int
    learning_rate: float
    epochs: int
    teacher_forcing_ratio: float
    gradient_clip: float
    dropout: float
    max_decode_length: int
    min_frequency: int
    num_workers: int
    device: str
    use_amp: bool
    resume_from_latest: bool
    limit_train_examples: int | None
    limit_validation_examples: int | None
    limit_test_examples: int | None
    run_test: bool


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a GRU seq2seq tashkeel model.")
    parser.add_argument("--train-path", default="output_strict/train.jsonl")
    parser.add_argument("--validation-path", default="output_strict/validation.jsonl")
    parser.add_argument("--test-path", default="output_strict/test.jsonl")
    parser.add_argument("--checkpoint-dir", default="artifacts/gru_seq2seq")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--encoder-hidden-size", type=int, default=256)
    parser.add_argument("--decoder-hidden-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-decode-length", type=int, default=420)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--resume-from-latest", action="store_true", help="Resume from latest.pt if present.")
    parser.add_argument("--limit-train-examples", type=int, default=None)
    parser.add_argument("--limit-validation-examples", type=int, default=None)
    parser.add_argument("--limit-test-examples", type=int, default=None)
    parser.add_argument("--run-test", action="store_true")
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def maybe_limit_examples(examples, limit: int | None):
    if limit is None:
        return examples
    return examples[:limit]


def build_dataloaders(config: TrainingConfig):
    train_examples = maybe_limit_examples(load_jsonl(Path(config.train_path)), config.limit_train_examples)
    validation_examples = maybe_limit_examples(load_jsonl(Path(config.validation_path)), config.limit_validation_examples)
    test_examples = maybe_limit_examples(load_jsonl(Path(config.test_path)), config.limit_test_examples)

    vocab = build_char_vocab(train_examples, min_frequency=config.min_frequency)
    train_dataset = TashkeelDataset(train_examples, vocab)
    validation_dataset = TashkeelDataset(validation_examples, vocab)
    test_dataset = TashkeelDataset(test_examples, vocab)
    collate_fn = make_collate_fn(vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    return vocab, train_loader, validation_loader, test_loader


def build_model(config: TrainingConfig, vocab_size: int, pad_id: int, bos_id: int, eos_id: int) -> GRUSeq2Seq:
    return GRUSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        encoder_hidden_size=config.encoder_hidden_size,
        decoder_hidden_size=config.decoder_hidden_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        dropout=config.dropout,
    )


def train_one_epoch(
    model: GRUSeq2Seq,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
    gradient_clip: float,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        source_ids = batch["source_ids"].to(device)
        source_lengths = batch["source_lengths"].to(device)
        target_ids = batch["target_ids"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(
                source_ids=source_ids,
                source_lengths=source_lengths,
                target_ids=target_ids,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            targets = target_ids[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: GRUSeq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    vocab,
    device: torch.device,
    max_decode_length: int,
    use_amp: bool,
) -> dict[str, object]:
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

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(
                source_ids=source_ids,
                source_lengths=source_lengths,
                target_ids=target_ids,
                teacher_forcing_ratio=0.0,
            )
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


def save_checkpoint(
    checkpoint_path: Path,
    model: GRUSeq2Seq,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def load_history(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def train_model(config: TrainingConfig) -> dict[str, object]:
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vocab, train_loader, validation_loader, test_loader = build_dataloaders(config)
    save_vocab(vocab, checkpoint_dir / "vocab.json")
    (checkpoint_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    device = torch.device(config.device)
    model = build_model(config, len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history_path = checkpoint_dir / "history.json"
    history: list[dict[str, object]] = []
    best_validation_cer = float("inf")
    best_epoch = -1
    start_epoch = 1

    if config.resume_from_latest and (checkpoint_dir / "latest.pt").exists():
        latest_state = torch.load(checkpoint_dir / "latest.pt", map_location=device)
        model.load_state_dict(latest_state["model_state_dict"])
        optimizer.load_state_dict(latest_state["optimizer_state_dict"])
        start_epoch = int(latest_state["epoch"]) + 1
        saved_history = load_history(history_path)
        if isinstance(saved_history, dict):
            history = list(saved_history.get("history", []))
            best_validation_cer = float(saved_history.get("best_validation_cer", best_validation_cer))
            best_epoch = int(saved_history.get("best_epoch", best_epoch))
        if (checkpoint_dir / "best.pt").exists() and best_epoch < 0:
            best_state = torch.load(checkpoint_dir / "best.pt", map_location=device)
            best_epoch = int(best_state.get("epoch", best_epoch))
            best_metrics = best_state.get("metrics", {})
            if isinstance(best_metrics, dict):
                try:
                    best_validation_cer = float(best_metrics["validation_metrics"]["overall"]["diacritic_cer"])
                except Exception:
                    pass

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=config.teacher_forcing_ratio,
            gradient_clip=config.gradient_clip,
            scaler=scaler,
            use_amp=use_amp,
        )
        validation = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            vocab=vocab,
            device=device,
            max_decode_length=config.max_decode_length,
            use_amp=use_amp,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_metrics": validation["metrics"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, epoch_metrics)
        validation_cer = validation["metrics"]["overall"]["diacritic_cer"]
        if validation_cer < best_validation_cer:
            best_validation_cer = validation_cer
            best_epoch = epoch
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, epoch_metrics)

    result = {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_cer": best_validation_cer,
    }

    if config.run_test:
        best_state = torch.load(checkpoint_dir / "best.pt", map_location=device)
        model.load_state_dict(best_state["model_state_dict"])
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            vocab=vocab,
            device=device,
            max_decode_length=config.max_decode_length,
            use_amp=use_amp,
        )
        result["test"] = test_metrics
        print(json.dumps({"test": test_metrics}, ensure_ascii=False))

    history_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    config = parse_args()
    train_model(config)


if __name__ == "__main__":
    main()
