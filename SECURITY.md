# Security policy

## Reporting

Do not publish vulnerabilities, credentials, unsafe model-loading behavior, or data
leaks in a public issue. Use GitHub's private vulnerability reporting for
`BangProx/OPD-study`. Include affected version, reproduction, impact, and a suggested
mitigation if available. Expect an acknowledgement target of seven days for this
volunteer pre-release project.

## Supported version

Only the latest `main` revision is supported before 1.0.

## Security boundaries

- Core code never executes downloaded model code (`trust_remote_code=false`).
- Research assets require pinned revisions and explicit license/download acceptance.
- Checkpoint loading is fixed to PyTorch `weights_only=True`; model configs are validated
  primitive mappings. Still load only artifacts produced by a trusted local run.
- HTML reports escape prompts and completions.
- API keys are neither required by core nor stored in config.
- A failed quantization probe never falls back to full fine-tuning.

Research models can generate unsafe or incorrect text. The project is educational and
does not make model outputs safe for deployment.
