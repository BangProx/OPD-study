"""A transparent character tokenizer for the offline arithmetic course."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from opd_study.data.tiny_arithmetic import ArithmeticExample
from opd_study.types import TrajectoryBatch


@dataclass(frozen=True)
class CharacterTokenizer:
    """Fixed-vocabulary tokenizer with no files, network calls or hidden template."""

    symbols: str = (
        "0123456789+-*/()=:,.!?_ "
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "\n"
    )

    def __post_init__(self) -> None:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must not contain duplicates")

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def unk_token_id(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return 4 + len(self.symbols)

    @property
    def _character_to_id(self) -> dict[str, int]:
        return {character: index + 4 for index, character in enumerate(self.symbols)}

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        table = self._character_to_id
        ids = [table.get(character, self.unk_token_id) for character in text]
        if bos:
            ids.insert(0, self.bos_token_id)
        if eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        id_to_character = {
            index + 4: character for index, character in enumerate(self.symbols)
        }
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                if skip_special_tokens:
                    continue
                pieces.append({0: "<pad>", 1: "<bos>", 2: "<eos>"}[token_id])
            elif token_id == self.unk_token_id:
                pieces.append("�" if skip_special_tokens else "<unk>")
            else:
                pieces.append(id_to_character.get(token_id, "�"))
        return "".join(pieces)


def collate_examples(
    examples: Sequence[ArithmeticExample],
    tokenizer: CharacterTokenizer,
    *,
    max_length: int = 128,
    device: torch.device | str | None = None,
) -> TrajectoryBatch:
    """Encode demonstrations and mark only response tokens (including EOS) for loss."""

    if not examples:
        raise ValueError("cannot collate an empty example sequence")
    encoded: list[list[int]] = []
    response_masks: list[list[bool]] = []
    prompt_lengths: list[int] = []
    for example in examples:
        prompt_ids = tokenizer.encode(example.prompt, bos=True)
        response_ids = tokenizer.encode(example.response, eos=True)
        combined = prompt_ids + response_ids
        if len(combined) > max_length:
            raise ValueError(
                f"example {example.example_id} needs {len(combined)} tokens, "
                f"exceeding max_length={max_length}"
            )
        encoded.append(combined)
        response_masks.append([False] * len(prompt_ids) + [True] * len(response_ids))
        prompt_lengths.append(len(prompt_ids))

    padded_length = max(len(ids) for ids in encoded)
    token_ids = torch.full(
        (len(examples), padded_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    response_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for row_index, (ids, row_response_mask) in enumerate(
        zip(encoded, response_masks, strict=False)
    ):
        length = len(ids)
        token_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_index, :length] = True
        response_mask[row_index, :length] = torch.tensor(
            row_response_mask, dtype=torch.bool, device=device
        )
    return TrajectoryBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
    )


def collate_multiturn_text(
    rows: Sequence[tuple[str, Sequence[str]]],
    tokenizer: CharacterTokenizer,
    *,
    max_length: int = 128,
    device: torch.device | str | None = None,
) -> TrajectoryBatch:
    """Encode prompt + textual turns while preserving turn/step IDs per token."""

    if not rows:
        raise ValueError("cannot collate an empty multi-turn sequence")
    encoded_rows: list[list[int]] = []
    response_rows: list[list[bool]] = []
    turn_rows: list[list[int]] = []
    prompt_lengths: list[int] = []
    for prompt, turns in rows:
        if not turns:
            raise ValueError("every multi-turn row needs at least one response turn")
        prompt_ids = tokenizer.encode(prompt, bos=True)
        ids = list(prompt_ids)
        response = [False] * len(prompt_ids)
        turn_ids = [-1] * len(prompt_ids)
        for turn_id, turn in enumerate(turns):
            turn_tokens = tokenizer.encode(turn + "\n")
            ids.extend(turn_tokens)
            response.extend([True] * len(turn_tokens))
            turn_ids.extend([turn_id] * len(turn_tokens))
        ids.append(tokenizer.eos_token_id)
        response.append(True)
        turn_ids.append(len(turns) - 1)
        if len(ids) > max_length:
            raise ValueError(f"multi-turn row needs {len(ids)} tokens; max_length={max_length}")
        encoded_rows.append(ids)
        response_rows.append(response)
        turn_rows.append(turn_ids)
        prompt_lengths.append(len(prompt_ids))

    padded_length = max(len(ids) for ids in encoded_rows)
    shape = (len(rows), padded_length)
    token_ids = torch.full(shape, tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros(shape, dtype=torch.bool, device=device)
    response_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    turn_ids_tensor = torch.full(shape, -1, dtype=torch.long, device=device)
    for index, (ids, response, turn_ids) in enumerate(
        zip(encoded_rows, response_rows, turn_rows, strict=False)
    ):
        length = len(ids)
        token_ids[index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention[index, :length] = True
        response_mask[index, :length] = torch.tensor(
            response, dtype=torch.bool, device=device
        )
        turn_ids_tensor[index, :length] = torch.tensor(
            turn_ids, dtype=torch.long, device=device
        )
    return TrajectoryBatch(
        token_ids=token_ids,
        attention_mask=attention,
        response_mask=response_mask,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        turn_ids=turn_ids_tensor,
        step_ids=turn_ids_tensor.clone(),
        terminal=torch.ones(len(rows), dtype=torch.bool, device=device),
    )
