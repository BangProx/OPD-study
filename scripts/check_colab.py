"""Validate the default-safe and optional-Qwen contracts of the Colab notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat

REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> int:
    path = REPOSITORY / "notebooks/colab/quickstart.ipynb"
    notebook = nbformat.read(path, as_version=4)
    metadata = notebook.metadata["opd_study"]
    assert metadata["lesson_id"] == "COLAB-QUICKSTART"
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    setup_cell = next(
        cell
        for cell in code_cells
        if cell.metadata["opd_study"]["cell_id"] == "COLAB-S02"
    )
    assert "sys.path.insert(0, str(repo_root / 'src'))" in setup_cell.source
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    source = "\n".join(cell.source for cell in code_cells)
    assert "RUN_OPTIONAL_QWEN_LORA = False" in source
    assert "RUN_OPTIONAL_QWEN_QLORA = False" in source
    assert "configs/laptop/gsm8k_lora.yaml" in source
    assert "configs/laptop/gsm8k_qlora.yaml" in source
    assert "--accept-dataset-license" in source
    assert "--accept-model-license" in source
    assert "torch.cuda.is_available()" in source
    assert "distribution_diagnostics.png" in source
    assert "checkpoints/optimizer.pt" in source
    assert "tensorboard" in source
    outputs = "\n".join(
        str(output.get("text", ""))
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert "SKIPPED:" in outputs
    assert "model-download-free" in outputs
    assert "/Users/" not in outputs and "Documents/nlp" not in outputs
    print("validated Colab default-safe path and opt-in LoRA/QLoRA contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
