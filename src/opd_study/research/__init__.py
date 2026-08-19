"""Optional Hugging Face adapters; importing core never imports heavy frameworks."""

from opd_study.research.preflight import ResearchPreflight, research_preflight
from opd_study.research.training import extract_gsm8k_answer, run_research_smoke

__all__ = [
    "ResearchPreflight",
    "extract_gsm8k_answer",
    "research_preflight",
    "run_research_smoke",
]
