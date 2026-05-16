from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models import GRUSeq2Seq, SimpleCharSeq2Seq
from vocab import CharVocabulary, load_vocab


SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[\.\!\?؟؛:\n])")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    vocab: CharVocabulary
    config: dict[str, Any]
    model_kind: str
    checkpoint_path: Path
    device: torch.device


def detect_model_kind(config: dict[str, Any]) -> str:
    if "encoder_hidden_size" in config and "decoder_hidden_size" in config:
        return "attention"
    if "hidden_size" in config:
        return "simple"
    raise ValueError("Unsupported checkpoint config format.")


def choose_checkpoint(artifact_dir: Path) -> Path:
    best_path = artifact_dir / "best.pt"
    latest_path = artifact_dir / "latest.pt"
    if best_path.exists():
        return best_path
    if latest_path.exists():
        return latest_path
    raise FileNotFoundError(f"No best.pt or latest.pt found in {artifact_dir}")


def build_model(config: dict[str, Any], vocab: CharVocabulary, model_kind: str) -> torch.nn.Module:
    if model_kind == "attention":
        return GRUSeq2Seq(
            vocab_size=len(vocab),
            embedding_dim=int(config["embedding_dim"]),
            encoder_hidden_size=int(config["encoder_hidden_size"]),
            decoder_hidden_size=int(config["decoder_hidden_size"]),
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            dropout=float(config.get("dropout", 0.0)),
        )
    if model_kind == "simple":
        return SimpleCharSeq2Seq(
            vocab_size=len(vocab),
            embedding_dim=int(config["embedding_dim"]),
            hidden_size=int(config["hidden_size"]),
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            dropout=0.0,
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def load_artifact(artifact_dir: str | Path, device: str | torch.device | None = None) -> LoadedModel:
    artifact_path = Path(artifact_dir).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_path}")

    config_path = artifact_path / "config.json"
    vocab_path = artifact_path / "vocab.json"
    checkpoint_path = choose_checkpoint(artifact_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {artifact_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab.json in {artifact_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab = load_vocab(vocab_path)
    model_kind = detect_model_kind(config)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(config, vocab, model_kind)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    return LoadedModel(
        model=model,
        vocab=vocab,
        config=config,
        model_kind=model_kind,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
    )


def split_text_into_chunks(text: str, max_chunk_chars: int) -> list[str]:
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive.")

    sentence_like_parts = [part for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part]
    if not sentence_like_parts:
        return []

    chunks: list[str] = []
    current = ""

    for part in sentence_like_parts:
        if len(part) > max_chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_overlong_chunk(part, max_chunk_chars))
            continue
        if current and len(current) + len(part) > max_chunk_chars:
            chunks.append(current)
            current = part
        else:
            current += part

    if current:
        chunks.append(current)
    return chunks


def _split_overlong_chunk(text: str, max_chunk_chars: int) -> list[str]:
    tokens = re.findall(r"\s+|\S+\s*", text)
    chunks: list[str] = []
    current = ""

    for token in tokens:
        if len(token) > max_chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(token[index : index + max_chunk_chars] for index in range(0, len(token), max_chunk_chars))
            continue
        if current and len(current) + len(token) > max_chunk_chars:
            chunks.append(current)
            current = token
        else:
            current += token

    if current:
        chunks.append(current)
    return chunks


@torch.no_grad()
def _predict_chunk(loaded: LoadedModel, text: str, max_decode_length: int) -> str:
    source_ids = loaded.vocab.encode(text, add_bos=False, add_eos=True)
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=loaded.device)
    source_lengths = torch.tensor([len(source_ids)], dtype=torch.long, device=loaded.device)
    decoded = loaded.model.greedy_decode(source_tensor, source_lengths, max_decode_length=max_decode_length)
    return loaded.vocab.decode(decoded[0].tolist())


@torch.no_grad()
def predict_text(
    loaded: LoadedModel,
    text: str,
    max_decode_length: int | None = None,
    max_chunk_chars: int | None = None,
) -> str:
    decode_limit = max_decode_length or int(loaded.config.get("max_decode_length", 420))
    chunk_limit = max_chunk_chars or int(loaded.config.get("inference_chunk_chars", 180))
    chunks = split_text_into_chunks(text, max_chunk_chars=chunk_limit)
    if not chunks:
        return ""
    return "".join(_predict_chunk(loaded, chunk, max_decode_length=decode_limit) for chunk in chunks)
