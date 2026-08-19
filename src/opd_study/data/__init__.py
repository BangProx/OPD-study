"""Offline datasets and tokenization for the educational mini backend."""

from opd_study.data.gsm8k import fetch_gsm8k, gsm8k_to_mini_examples, load_gsm8k_rows
from opd_study.data.tiny_arithmetic import (
    ArithmeticExample,
    TinyArithmeticSplits,
    generate_tiny_arithmetic,
)
from opd_study.data.tokenizer import (
    CharacterTokenizer,
    collate_examples,
    collate_multiturn_text,
)

__all__ = [
    "ArithmeticExample",
    "CharacterTokenizer",
    "TinyArithmeticSplits",
    "collate_examples",
    "collate_multiturn_text",
    "fetch_gsm8k",
    "generate_tiny_arithmetic",
    "gsm8k_to_mini_examples",
    "load_gsm8k_rows",
]
