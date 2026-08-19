"""A readable decoder-only Transformer small enough for CPU course exercises."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from opd_study.masks import causal_attention_mask


@dataclass(frozen=True)
class TinyTransformerConfig:
    """Architecture-only configuration; training choices live elsewhere."""

    vocab_size: int
    max_sequence_length: int = 128
    number_of_layers: int = 2
    hidden_size: int = 64
    number_of_heads: int = 4
    feed_forward_size: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.vocab_size,
            self.max_sequence_length,
            self.number_of_layers,
            self.hidden_size,
            self.number_of_heads,
            self.feed_forward_size,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("all size/count fields must be positive")
        if self.hidden_size % self.number_of_heads != 0:
            raise ValueError("hidden_size must be divisible by number_of_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        """Return a weights-only-safe checkpoint representation."""

        return {
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "number_of_layers": self.number_of_layers,
            "hidden_size": self.hidden_size,
            "number_of_heads": self.number_of_heads,
            "feed_forward_size": self.feed_forward_size,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TinyTransformerConfig:
        """Validate a primitive checkpoint mapping without arbitrary unpickling."""

        expected = {
            "vocab_size",
            "max_sequence_length",
            "number_of_layers",
            "hidden_size",
            "number_of_heads",
            "feed_forward_size",
            "dropout",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            unknown = sorted(set(value) - expected)
            raise ValueError(f"invalid model config keys; missing={missing}, unknown={unknown}")
        integer_names = (
            "vocab_size",
            "max_sequence_length",
            "number_of_layers",
            "hidden_size",
            "number_of_heads",
            "feed_forward_size",
        )
        integers: dict[str, int] = {}
        for name in integer_names:
            item = value[name]
            if not isinstance(item, int) or isinstance(item, bool):
                raise TypeError(f"model config {name} must be an integer")
            integers[name] = item
        dropout = value["dropout"]
        if not isinstance(dropout, int | float) or isinstance(dropout, bool):
            raise TypeError("model config dropout must be numeric")
        return cls(**integers, dropout=float(dropout))

    @classmethod
    def student(
        cls, *, vocab_size: int, max_sequence_length: int = 128
    ) -> TinyTransformerConfig:
        return cls(vocab_size=vocab_size, max_sequence_length=max_sequence_length)

    @classmethod
    def teacher(
        cls, *, vocab_size: int, max_sequence_length: int = 128
    ) -> TinyTransformerConfig:
        return cls(
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            number_of_layers=4,
            hidden_size=128,
            number_of_heads=4,
            feed_forward_size=512,
        )


class TinyCausalLM(nn.Module):
    """Minimal causal LM with learned positions and explicit padding/causal masks."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length, config.hidden_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.number_of_heads,
            dim_feedforward=config.feed_forward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.number_of_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        _, sequence_length = token_ids.shape
        if sequence_length < 1 or sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length must be within [1, {self.config.max_sequence_length}], "
                f"got {sequence_length}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        if attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must have the same shape as token_ids")
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) * math.sqrt(self.config.hidden_size)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = self.transformer(
            hidden,
            mask=causal_attention_mask(sequence_length, device=token_ids.device),
            src_key_padding_mask=~attention_mask.bool(),
        )
        return self.lm_head(self.final_norm(hidden))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int,
        min_new_tokens: int = 0,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively extend an unpadded ``[1, T]`` prompt."""

        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("the mini generator expects one unpadded prompt [1, T]")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not 0 <= min_new_tokens <= max_new_tokens:
            raise ValueError("min_new_tokens must be within [0, max_new_tokens]")
        if temperature < 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and non-negative")
        was_training = self.training
        self.eval()
        generated = prompt_ids
        try:
            for generated_count in range(1, max_new_tokens + 1):
                if generated.shape[1] >= self.config.max_sequence_length:
                    break
                logits = self(generated)[:, -1]
                if temperature == 0.0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
                    if generator is not None and probabilities.device.type != "cpu":
                        # A CPU generator gives reproducible sampling on CUDA/MPS too.
                        next_token = torch.multinomial(
                            probabilities.cpu(), num_samples=1, generator=generator
                        ).to(probabilities.device)
                    else:
                        next_token = torch.multinomial(
                            probabilities, num_samples=1, generator=generator
                        )
                generated = torch.cat((generated, next_token), dim=1)
                if next_token.item() == eos_token_id and generated_count >= min_new_tokens:
                    break
        finally:
            self.train(was_training)
        return generated

    def number_of_parameters(self, *, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (
                parameter for parameter in parameters if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)
