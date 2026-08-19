# Contributing

Thanks for improving OPD-study. Small, source-backed changes are preferred.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev,notebooks,research]"
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
python scripts/check_notebooks.py --require-executed
```

## Course changes

Edit Korean content first in `scripts/build_notebooks.py`, add/update tests, regenerate
both languages, execute both tracks, then run the parity checker. Do not hand-edit one
generated notebook. Keep code cells byte-identical and translate meaning, not word order.

Every lesson needs a prediction, executable checks, an optional exercise, two recurring
mistake notes, source IDs, a map position, and a 60-second summary.

## Algorithm changes

1. Add a primary paper/version and any official code/license to `docs/sources.yml`.
2. State whether code is copied, adapted, behaviorally compared, or clean-room.
3. Name teacher/student distributions and tensor shapes in the public API.
4. Test analytic values, masks, gradient ownership, and edge cases.
5. Label paper-faithful behavior separately from mini-backend simplifications.

Do not cherry-pick favorable seeds or delete results where SFT wins. Never commit model
weights, datasets, tokens, credentials, TensorBoard runs, or generated checkpoints.

## Pull requests

Use a focused branch and explain motivation, evidence, commands run, runtime, and
limitations. By contributing, you agree that your contribution is licensed under
Apache-2.0.
