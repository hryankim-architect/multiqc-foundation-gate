# `scripts/`

Operational helpers, not the pipeline itself.

- `run_lab.sh`, one-liner to invoke `make run` on a Polish-Phase5 lab node
  with the substrate env vars set to lab defaults.
- `check_english_only.py`, local pre-commit hook that blocks commits adding CJK characters to
  public artifacts (R6 convention).

To skip the English-only scan for an intentionally bilingual file (such as
a translation appendix), add the path to `scripts/english-only.skip`, one
path per line.
