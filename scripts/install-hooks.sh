#!/usr/bin/env bash
# Install a local pre-commit hook that runs the English-only (CJK) scanner
# before each commit. Run once per clone:
#
#     bash scripts/install-hooks.sh
#
set -e
root="$(git rev-parse --show-toplevel)"
hook="$root/.git/hooks/pre-commit"
cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Block a commit if it would put CJK characters into public artifacts (R6).
python scripts/check_english_only.py || {
  echo "pre-commit: CJK characters found in public artifacts; commit aborted." >&2
  echo "Move non-English content to private notes, or add the path to scripts/english-only.skip." >&2
  exit 1
}
HOOK
chmod +x "$hook"
echo "Installed pre-commit hook -> $hook"
