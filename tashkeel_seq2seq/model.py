from __future__ import annotations

import random

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_encoder = self.encoder_projection(encoder_outputs)
        projected_decoder = self.decoder_projection(decoder_hidden).unsqueeze(1)
        energy = torch.tanh(projected_encoder + projected_decoder)
        scores = self.score(energy).squeeze(-1)
        scores = scores.masked_fill(~source_mask, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class GRUSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.decoder_hidden_size = decoder_hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=encoder_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder = nn.GRU(
            input_size=embedding_dim + encoder_hidden_size * 2,
            hidden_size=decoder_hidden_size,
            batch_first=True,
        )
        self.attention = AdditiveAttention(
            encoder_dim=encoder_hidden_size * 2,
            decoder_dim=decoder_hidden_size,
            attention_dim=decoder_hidden_size,
        )
        self.encoder_to_decoder = nn.Linear(encoder_hidden_size * 2, decoder_hidden_size)
        self.output_projection = nn.Linear(decoder_hidden_size + encoder_hidden_size * 2 + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, source_ids: torch.Tensor, source_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        packed_outputs, hidden = self.encoder(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        decoder_hidden = torch.tanh(self.encoder_to_decoder(hidden)).unsqueeze(0)
        return encoder_outputs, decoder_hidden

    def _decode_step(
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

    def forward(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        encoder_outputs, hidden = self.encode(source_ids, source_lengths)
        batch_size = source_ids.size(0)
        target_steps = target_ids.size(1) - 1
        vocab_size = self.output_projection.out_features
        logits = torch.zeros(batch_size, target_steps, vocab_size, device=source_ids.device)
        source_mask = source_ids.ne(self.pad_id)
        decoder_input = target_ids[:, 0]

        for step in range(target_steps):
            step_logits, hidden = self._decode_step(decoder_input, hidden, encoder_outputs, source_mask)
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
        encoder_outputs, hidden = self.encode(source_ids, source_lengths)
        batch_size = source_ids.size(0)
        source_mask = source_ids.ne(self.pad_id)
        decoder_input = torch.full((batch_size,), self.bos_id, dtype=torch.long, device=source_ids.device)
        generated = []

        for _ in range(max_decode_length):
            step_logits, hidden = self._decode_step(decoder_input, hidden, encoder_outputs, source_mask)
            decoder_input = step_logits.argmax(dim=-1)
            generated.append(decoder_input)
            if torch.all(decoder_input.eq(self.eos_id)):
                break

        if not generated:
            return torch.empty(batch_size, 0, dtype=torch.long, device=source_ids.device)
        return torch.stack(generated, dim=1)

