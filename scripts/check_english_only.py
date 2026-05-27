#!/usr/bin/env python3
"""Fail if any CJK characters appear in tracked, public artifacts.

The R6 convention is: internal docs may be Korean+English, but every public
GitHub artifact (code, README, lessons-learned, architecture docs) must be
English-only. This scanner enforces that as a CI gate.

Paths scanned by default:
    README.md, src/**/*.py, tests/**/*.py, docs/**/*.md, audit/**/*.md,
    scripts/**/*.sh, scripts/**/*.py, .github/workflows/**/*.yml

Skip the scan for a file by listing it in scripts/english-only.skip
(one path per line).

Also runnable as a pre-push client-side gate from any commit_*.sh helper
(Polish-Phase5-Lτ). Catches CJK pre-push so we don't waste a CI round-trip.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

CJK_RANGES = [
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK unified ideographs ext A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF00, 0xFFEF),  # Halfwidth/fullwidth
]

DEFAULT_GLOBS = [
    "README.md",
    "src/**/*.py",
    "tests/**/*.py",
    "docs/**/*.md",
    "audit/**/*.md",
    "scripts/**/*.sh",
    "scripts/**/*.py",
    ".github/workflows/**/*.yml",
]


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in CJK_RANGES)


def load_skip(repo_root: Path) -> set[str]:
    skip_file = repo_root / "scripts" / "english-only.skip"
    if not skip_file.exists():
        return set()
    return {
        line.strip()
        for line in skip_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def gather_files(repo_root: Path, skip: set[str]) -> list[Path]:
    out: set[Path] = set()
    for pattern in DEFAULT_GLOBS:
        for match in repo_root.glob(pattern):
            if match.is_file():
                rel = match.relative_to(repo_root).as_posix()
                if rel not in skip:
                    out.add(match)
    return sorted(out)


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return (lineno, char, line) for each CJK occurrence."""
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return findings
    for i, line in enumerate(text.splitlines(), start=1):
        for ch in line:
            if is_cjk(ch):
                findings.append((i, ch, line.rstrip()))
                break
    return findings


def main() -> int:
    repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()
    skip = load_skip(repo_root)
    files = gather_files(repo_root, skip)

    bad: list[tuple[Path, list[tuple[int, str, str]]]] = []
    for path in files:
        findings = scan_file(path)
        if findings:
            bad.append((path, findings))

    if not bad:
        print(f"OK: scanned {len(files)} public artifacts; no CJK characters found.")
        return 0

    print("FAIL: CJK characters found in public artifacts:\n")
    for path, findings in bad:
        rel = path.relative_to(repo_root).as_posix()
        for lineno, ch, line in findings[:5]:
            print(f"  {rel}:{lineno}: {ch!r}  {line[:120]}")
        if len(findings) > 5:
            print(f"  ... and {len(findings) - 5} more occurrences in {rel}")
        print()
    print(
        "Internal docs may be Korean+English (R6), but every public artifact must be "
        "English-only.\nMove non-English content to private docs or add the path to "
        "scripts/english-only.skip if the file is intentionally bilingual."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
