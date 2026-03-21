#!/usr/bin/env python3
"""
WER Accuracy Report Generator
Compares transcription output against reference text and produces a report.

Usage:
    python wer_report.py --hypothesis logs/transcription_YYYYMMDD_HHMMSS.txt \
                         --reference  reference.txt

    # Or evaluate against all log files:
    python wer_report.py --all
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime


# ── WER calculation ───────────────────────────────────────────────────────────

def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def wer(reference: list[str], hypothesis: list[str]) -> dict:
    """
    Compute Word Error Rate via dynamic programming (Wagner-Fischer).
    WER = (S + D + I) / N
        S = substitutions, D = deletions, I = insertions, N = ref length
    """
    r, h = reference, hypothesis
    n, m = len(r), len(h)

    # Build cost matrix
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost, # substitution
            )

    # Back-trace to count S / D / I
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if r[i - 1] == h[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                if cost:
                    S += 1
                i -= 1; j -= 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1; i -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            I += 1; j -= 1

    rate = (S + D + I) / max(n, 1) * 100
    return {
        "wer_pct": round(rate, 2),
        "substitutions": S,
        "deletions": D,
        "insertions": I,
        "ref_words": n,
        "hyp_words": m,
        "edit_distance": d[n][m],
    }


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    r = list(reference.replace(" ", ""))
    h = list(hypothesis.replace(" ", ""))
    n = len(r)
    m = len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return round(d[n][m] / max(n, 1) * 100, 2)


# ── Report formatting ─────────────────────────────────────────────────────────

def rating(wer_pct: float) -> str:
    if wer_pct < 5:   return "🏆 Excellent"
    if wer_pct < 10:  return "✅ Good"
    if wer_pct < 15:  return "🎯 Target met (< 15%)"
    if wer_pct < 25:  return "⚠️  Needs improvement"
    return "❌ Poor"


def print_report(result: dict, hyp_path: str, ref_path: str):
    w = result
    width = 60
    sep = "─" * width

    print(f"""
╔{'═' * width}╗
║{'  WER Accuracy Report':^{width}}║
╚{'═' * width}╝

  Hypothesis : {hyp_path}
  Reference  : {ref_path}
  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{sep}
  Word Error Rate (WER)  : {w['wer_pct']:>6.2f}%   {rating(w['wer_pct'])}
  Char Error Rate (CER)  : {w['cer_pct']:>6.2f}%
{sep}
  Reference words        : {w['ref_words']}
  Hypothesis words       : {w['hyp_words']}
  Substitutions          : {w['substitutions']}
  Deletions              : {w['deletions']}
  Insertions             : {w['insertions']}
  Edit distance          : {w['edit_distance']}
{sep}
  Week 2 target (WER < 15%) : {'PASS ✅' if w['wer_pct'] < 15 else 'FAIL ❌'}
{sep}
""")


def save_report(result: dict, out_path: Path):
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Report saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def strip_timestamps(text: str) -> str:
    """Remove [HH:MM:SS] prefixes from transcription log lines."""
    return re.sub(r"\[\d{2}:\d{2}:\d{2}\]\s*", "", text)


def run_report(hyp_path: str, ref_path: str):
    hyp_raw = load_text(hyp_path)
    ref_raw = load_text(ref_path)

    hyp_clean = strip_timestamps(hyp_raw)
    ref_norm  = normalise(ref_raw)
    hyp_norm  = normalise(hyp_clean)

    result = wer(ref_norm, hyp_norm)
    result["cer_pct"] = cer(
        ref_raw.lower(), hyp_clean.lower()
    )
    result["hypothesis_file"] = hyp_path
    result["reference_file"]  = ref_path
    result["timestamp"]       = datetime.now().isoformat()

    print_report(result, hyp_path, ref_path)

    out = Path(hyp_path).with_suffix(".wer.json")
    save_report(result, out)
    return result


def run_all():
    log_dir = Path("logs")
    txt_files = sorted(log_dir.glob("transcription_*.txt"))
    ref_file  = Path("reference.txt")

    if not ref_file.exists():
        print("⚠️  reference.txt not found. Creating a sample reference file…")
        ref_file.write_text(
            "This is a sample reference sentence for testing word error rate. "
            "Please replace this file with your actual reference transcript.\n"
        )
        print(f"   Created: {ref_file}\n")

    if not txt_files:
        print("No transcription logs found in ./logs/")
        return

    print(f"Found {len(txt_files)} log file(s).\n")
    for f in txt_files:
        print(f"{'=' * 60}")
        run_report(str(f), str(ref_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute WER for transcription logs")
    parser.add_argument("--hypothesis", "-H", help="Path to transcription log (.txt)")
    parser.add_argument("--reference",  "-R", help="Path to reference text (.txt)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all log files in ./logs/ against reference.txt")
    args = parser.parse_args()

    if args.all:
        run_all()
    elif args.hypothesis and args.reference:
        run_report(args.hypothesis, args.reference)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python wer_report.py --hypothesis logs/transcription_20240101_120000.txt --reference reference.txt")
        print("  python wer_report.py --all")