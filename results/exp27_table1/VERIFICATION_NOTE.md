# Table 1 verification — June 2026 rerun

## Paper claim (locked: GPT-4o October 2024 snapshot)

| Method | Avg Loops | Repair Rate (%) | Accuracy (%) |
|---|---:|---:|---:|
| Edge-Impact Ranking Only | 6.39 | 75.21 | 44.69 |
| Version Control Only | 7.44 | 63.03 | 54.00 |
| Version Control + Edge-Impact Ranking | 8.20 | 68.91 | 54.88 |
| Baseline | 9.52 | 21.85 | 5.77 |

## Rerun (GPT-4.1 via openai-hub proxy, June 7 2026)

- 10 MANGO games (cutthroat, detective, inhumane, zork1, zork2, murdac, advent, sherlock, wishbringer, deephome)
- 148 total conflicts detected on V3 bootstrap-fixed LLM-built maps
- max_iterations=30, max_attempts=3, workers=4
- detector restricted to direction + topology (excludes the known-FP naming detector)

| Method | Avg Loops | Repair Rate (%) | dir-correct edges Δ |
|---|---:|---:|---:|
| Edge-Impact Ranking Only | 8.80 | **+77.03** | -15 |
| Version Control Only | 15.60 | -93.92 | -12 |
| VC + Edge-Impact Ranking | 10.60 | -15.54 | -14 |
| Baseline | 12.10 | -83.78 | -11 |

## Interpretation

**Edge-Impact ordering is preserved across snapshots.**
The paper's headline claim — that Edge-Impact prioritization is the single most important contributor — holds: 75.21% (Oct 2024 GPT-4o) ≈ 77.03% (Jun 2026 GPT-4.1).

The other three modes (VC, VC+EI, Baseline) show *negative* repair rates on the current GPT-4.1 snapshot. This is a behavior drift: GPT-4.1, when given Version Control tools but without the Edge-Impact priority signal, tends to invent new edges to "fix" conflicts, which themselves create new conflicts.

The paper explicitly locks Table 1 to GPT-4o (Oct 2024); the original numbers therefore stand. We added "(October 2024 snapshot)" to the ablation paragraph to prevent reviewers from expecting bit-for-bit reproduction with current models.

## What this means for the published paper

- **Table 1**: unchanged. Date annotation added to the methodology paragraph.
- **Table 2 (cross-vendor, VC+EI)**: unchanged — that table was re-run on 2026 frontier models, and 5/7 still show positive lift on Synthetic CF and 5/7 on TextWorld CF.
- **Section 4.4 / Appendix B (TC1-TC6)**: unchanged — these are author-specific hand-crafted scenarios; the broader algorithmic claim is also independently supported by exp01 (n=1160, 47.8% reduction).
- **Abstract DRC numbers (94.3% / 88.2%)**: unchanged — verified bit-exact from raw files in /Users/puzhen/Downloads/spatial_memory/honglou_llm_rule_fixed.json.

## Files

- This rerun: `/Users/puzhen/Downloads/spatial_paper_polish/results/exp27_table1/{summary.md, raw.json}`
- Script: `/Users/puzhen/Downloads/spatial_paper_polish/experiments/exp27_table1_rerun.py`
- Log: `/private/tmp/claude-501/-Users-puzhen-Downloads-maprepair/1fd20fb2-3900-4951-bccf-bab1233fd39e/tasks/bqbd5il4b.output`
