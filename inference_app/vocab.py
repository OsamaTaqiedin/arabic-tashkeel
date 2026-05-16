from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


@dataclass
class CharVocabulary:
    tokens: list[str]

    def __post_init__(self) -> None:
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = dict(enumerate(self.tokens))

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(self.token_to_id.get(character, self.unk_id) for character in text)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        characters: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), UNK_TOKEN)
            if token in SPECIAL_TOKENS:
                continue
            characters.append(token)
        return "".join(characters)

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharVocabulary":
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("Invalid vocabulary payload.")
        return cls(tokens=tokens)


def load_vocab(path: str | Path) -> CharVocabulary:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return CharVocabulary.from_dict(payload)
