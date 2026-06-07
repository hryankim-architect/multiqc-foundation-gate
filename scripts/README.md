# `scripts/`

Operational helpers, not the pipeline itself.

- `run_lab.sh` — invoke `make run` on a lab node with substrate env vars
  set to lab defaults (`chi-mac-m:8081` / `chi-mac-m:5050`).
- `run_phase_c.py` — Phase C augmentation orchestration: 10 base SRR → 40
  augmented FASTQ pairs. Accepts `--plan`, `--do-json`, `--do-fastq`.
- `calibrate_gate.py` — confidence-ECE + Brier diagnostic on the pooled
  5-fold held-out predictions; writes `audit/gate_calibration.md`.
- `interpret_gate.py` — permutation-importance interpretability diagnostic
  over 28 named features; writes `audit/gate_feature_importance.md`.
