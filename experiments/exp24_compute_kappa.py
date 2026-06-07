"""Compute Cohen's kappa between subagent labels and human labels on
the validation sheet produced by exp24_human_validation_sample.py.

Parses the human's filled-in `Human label: <label>` lines, joins with
the subagent labels in `sample_index.json`, and reports:
  - confusion matrix (subagent vs human)
  - Cohen's kappa (with 95% CI via bootstrap)
  - per-bucket agreement
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VALID_LABELS = (
    "CORRECT", "WRONG_DIRECTION", "WRONG_DST", "SPURIOUS_PAIR",
    "HALLUCINATED_DST", "HALLUCINATED_SRC", "UNCERTAIN",
)


def parse_human_labels(md_path: Path) -> Dict[int, str]:
    text = md_path.read_text()
    # Sections start with "## Sample NNN"
    sections = re.split(r"^## Sample (\d+)\b.*?$", text, flags=re.MULTILINE)
    out: Dict[int, str] = {}
    # re.split gives [pre, idx1, body1, idx2, body2, ...]
    for i in range(1, len(sections), 2):
        idx = int(sections[i])
        body = sections[i + 1]
        m = re.search(r"\*\*Human label:\*\*\s*(\S+)", body)
        if not m:
            continue
        label = m.group(1).strip().strip("_").strip().upper()
        if label in {"_", "____", "_____"} or not label:
            continue
        if label in VALID_LABELS:
            out[idx] = label
    return out


def cohens_kappa(subagent: List[str], human: List[str]) -> float:
    """Standard Cohen's kappa over a list of paired labels."""
    assert len(subagent) == len(human)
    n = len(subagent)
    if n == 0:
        return 0.0
    labels = sorted(set(subagent) | set(human))
    po = sum(1 for s, h in zip(subagent, human) if s == h) / n
    # marginal probabilities
    sub_dist = Counter(subagent)
    hum_dist = Counter(human)
    pe = sum((sub_dist[l] * hum_dist[l]) / (n * n) for l in labels)
    if abs(1 - pe) < 1e-9:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(subagent: List[str], human: List[str], n_boot: int = 1000,
                  seed: int = 42) -> Tuple[float, float]:
    import random
    rng = random.Random(seed)
    n = len(subagent)
    kappas = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        kappas.append(cohens_kappa([subagent[i] for i in idxs],
                                     [human[i] for i in idxs]))
    kappas.sort()
    lo = kappas[int(0.025 * n_boot)]
    hi = kappas[int(0.975 * n_boot)]
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path,
                    default=Path("results/exp24/human_validation_sheet.md"))
    ap.add_argument("--index", type=Path,
                    default=Path("results/exp24/sample_index.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/exp24/kappa_report.md"))
    args = ap.parse_args()

    index = json.loads(args.index.read_text())
    subagent_labels = {row["idx"]: row["predicted_bucket"] for row in index}
    human_labels = parse_human_labels(args.sheet)

    n_total = len(index)
    n_labelled = len(human_labels)
    print(f"Total samples: {n_total}")
    print(f"Human-labelled: {n_labelled} ({100*n_labelled/n_total:.1f}%)")

    pairs = [(subagent_labels[idx], human_labels[idx])
              for idx in sorted(human_labels.keys())
              if idx in subagent_labels]
    if not pairs:
        print("No paired labels to evaluate. Please annotate the sheet first.")
        return 1
    subagent_list, human_list = zip(*pairs)
    subagent_list, human_list = list(subagent_list), list(human_list)

    kappa = cohens_kappa(subagent_list, human_list)
    lo, hi = bootstrap_ci(subagent_list, human_list)

    # Confusion matrix
    all_labels = sorted(set(subagent_list) | set(human_list))
    matrix: Dict[Tuple[str, str], int] = Counter(zip(subagent_list, human_list))

    md = [
        "# F1-F9 Taxonomy: Human Validation Report\n",
        f"Paired labels: **{len(pairs)}** of {n_total} samples\n",
        f"Cohen's κ: **{kappa:.3f}** (95% bootstrap CI: [{lo:.3f}, {hi:.3f}])\n",
        "## Interpretation\n",
        ("- κ ≥ 0.81: almost perfect agreement\n"
         "- 0.61 ≤ κ < 0.81: substantial agreement\n"
         "- 0.41 ≤ κ < 0.61: moderate agreement\n"
         "- 0.21 ≤ κ < 0.41: fair agreement\n"
         "- κ < 0.21: slight / no agreement\n"),
        "## Confusion matrix (rows = subagent, cols = human)\n",
    ]
    header = "| subagent \\ human | " + " | ".join(all_labels) + " | total |"
    sep = "|------|" + "|".join(["---:"] * (len(all_labels) + 1)) + "|"
    md.extend([header, sep])
    for s in all_labels:
        row = [f"| **{s}**"]
        for h in all_labels:
            row.append(str(matrix.get((s, h), 0)))
        row.append(str(sum(matrix.get((s, h), 0) for h in all_labels)))
        md.append(" | ".join(row) + " |")

    md.append("\n## Per-bucket agreement\n")
    md.append("| subagent bucket | samples | agreed | agreement % |")
    md.append("|------|------:|-------:|------------:|")
    for s in all_labels:
        n_s = sum(matrix.get((s, h), 0) for h in all_labels)
        n_agree = matrix.get((s, s), 0)
        if n_s:
            md.append(f"| {s} | {n_s} | {n_agree} | {100*n_agree/n_s:.1f}% |")

    args.out.write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out}")
    print(f"\nκ = {kappa:.3f}  (95% CI: [{lo:.3f}, {hi:.3f}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
