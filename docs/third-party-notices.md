# Third-party notices

No third-party source file is vendored or copied into `src/`. The project independently
implements paper equations and cites them through `docs/sources.yml`.

- Hugging Face TRL (Apache-2.0), pinned commit: behavioral cross-check for GKD beta
  boundaries only.
- TCOD official repository (Apache-2.0), pinned commit: semantic and architecture
  comparison; distributed code not copied.
- SOD official repository (Apache-2.0), pinned commit: semantic comparison after noting
  that paper v3 is newer; code not copied.
- NAVER OPD² repository (Apache-2.0), pinned commit: semantic comparison; code not copied.
- THUNLP OPD and the unavailable vOPD repository are not reuse sources because a usable
  code license was not established at audit time.

PyTorch, Matplotlib, TensorBoard, PyYAML, Jupyter, Transformers, PEFT, Accelerate,
datasets, and bitsandbytes are dependencies under their respective licenses. Model and
dataset assets are downloaded separately and are not redistributed here.
