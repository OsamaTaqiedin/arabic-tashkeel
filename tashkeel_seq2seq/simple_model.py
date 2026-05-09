from __future__ import annotations

import random

import torch
from torch import nn


class SimpleCharSeq2Seq(nn.Module):
    """A minimal character-level GRU encoder-decoder without attention."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, source_ids: torch.Tensor, source_lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(source_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, hidden = self.encoder(packed)
        return hidden

    def forward(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        hidden = self.encode(source_ids, source_lengths)
        batch_size = source_ids.size(0)
        target_steps = target_ids.size(1) - 1
        vocab_size = self.output_projection.out_features
        logits = torch.zeros(batch_size, target_steps, vocab_size, device=source_ids.device)
        decoder_input = target_ids[:, 0]

        for step in range(target_steps):
            embedded = self.dropout(self.embedding(decoder_input)).unsqueeze(1)
            decoder_output, hidden = self.decoder(embedded, hidden)
            step_logits = self.output_projection(decoder_output.squeeze(1))
            logits[:, step, :] = step_logits
            teacher_force = random.random() < teacher_forcing_ratio
            predicted = step_logits.argmax(dim=-1)
            decoder_input = target_ids[:, step + 1] if teacher_force else predicted

        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        max_decode_length: int,
    ) -> torch.Tensor:
        hidden = self.encode(source_ids, source_lengths)
        batch_size = source_ids.size(0)
        decoder_input = torch.full((batch_size,), self.bos_id, dtype=torch.long, device=source_ids.device)
        generated = []

        for _ in range(max_decode_length):
            embedded = self.embedding(decoder_input).unsqueeze(1)
            decoder_output, hidden = self.decoder(embedded, hidden)
            step_logits = self.output_projection(decoder_output.squeeze(1))
            decoder_input = step_logits.argmax(dim=-1)
            generated.append(decoder_input)
            if torch.all(decoder_input.eq(self.eos_id)):
                break

        if not generated:
            return torch.empty(batch_size, 0, dtype=torch.long, device=source_ids.device)
        return torch.stack(generated, dim=1)
