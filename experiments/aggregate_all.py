"""Aggregate all experiment results into one consolidated report.

Reads `results/expNN/...` and writes `results/FINAL_REPORT.md` and
`results/FINAL_REPORT.json`. Idempotent; safe to call while experiments
are still running.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path("results")


def _load_json(p: Path) -> Optional[dict | list]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def summarize_exp01() -> Optional[Dict]:
    raw = _load_json(ROOT / "exp01" / "raw.json")
    if not isinstance(raw, list) or not raw:
        return None
    by_err: Dict[str, list] = {}
    for r in raw:
        by_err.setdefault(r["err_type"], []).append(r)
    out: Dict = {"n": len(raw),
                 "mean_reduction": statistics.mean(r["reduction"] for r in raw),
                 "mean_truth_in_lca": statistics.mean(r["truth_in_lca_fraction"] for r in raw),
                 "per_err": {}}
    for err, group in by_err.items():
        out["per_err"][err] = {
            "n": len(group),
            "mean_reduction": statistics.mean(r["reduction"] for r in group),
            "mean_truth_in_lca": statistics.mean(r["truth_in_lca_fraction"] for r in group),
        }
    return out


def summarize_exp02() -> Optional[Dict]:
    raw = _load_json(ROOT / "exp02" / "raw.json")
    if not isinstance(raw, list) or not raw:
        return None
    rho = [r["spearman"] for r in raw if r.get("spearman") == r.get("spearman")]
    tau = [r["kendall"] for r in raw if r.get("kendall") == r.get("kendall")]
    return {"n": len(rho), "mean_spearman": statistics.mean(rho),
            "stdev_spearman": statistics.pstdev(rho),
            "mean_kendall": statistics.mean(tau)}


def summarize_exp03() -> Optional[Dict]:
    raw = _load_json(ROOT / "exp03" / "raw.json")
    if not isinstance(raw, list) or not raw:
        return None
    cohort: Dict = {}
    for r in raw:
        cohort.setdefault((r["family"], r["err_type"]), []).append(r)
    out: Dict = {"n": len(raw), "per_cohort": {}}
    for (f, e), group in cohort.items():
        n = len(group)
        out["per_cohort"][f"{f}__{e}"] = {
            "n": n,
            "heuristic_conflict_free_pct": 100 * sum(1 for r in group if r.get("heuristic_conflict_free")) / n,
            "heuristic_dir_acc_pct": 100 * statistics.mean(r.get("heuristic_gt_direction_accuracy", 0) or 0 for r in group),
            "random_conflict_free_pct": 100 * sum(1 for r in group if r.get("random_conflict_free")) / n,
            "random_dir_acc_pct": 100 * statistics.mean(r.get("random_gt_direction_accuracy", 0) or 0 for r in group),
        }
    return out


def summarize_exp04() -> Optional[Dict]:
    raw = _load_json(ROOT / "exp04" / "raw.json")
    if not isinstance(raw, list) or not raw:
        return None
    by_mode: Dict = {}
    for r in raw:
        if r.get("mode"):
            by_mode.setdefault(r["mode"], []).append(r)
    out: Dict = {"n": len(raw), "per_mode": {}}
    for mode, rs in by_mode.items():
        out["per_mode"][mode] = {
            "n": len(rs),
            "conflict_free_pct": 100 * sum(1 for r in rs if r.get("conflict_free")) / len(rs),
            "mean_edge_recall_pct": 100 * statistics.mean(r.get("gt_edge_recall", 0) or 0 for r in rs),
            "mean_dir_acc_pct": 100 * statistics.mean(r.get("gt_direction_accuracy", 0) or 0 for r in rs),
            "strict_correct_pct": 100 * sum(1 for r in rs
                                              if r.get("conflict_free")
                                              and r.get("gt_edge_recall", 0) >= 0.999
                                              and r.get("gt_direction_accuracy", 0) >= 0.999) / len(rs),
            "mean_iterations": statistics.mean(r.get("iterations", 0) or 0 for r in rs),
        }
    return out


def summarize_exp05(model: str = "gpt-4.1-mini") -> Optional[Dict]:
    model_dir = ROOT / "exp05" / model
    if not model_dir.exists():
        return None
    out: Dict = {"per_mode": {}}
    for mode in ("baseline", "edge_impact", "vc_only", "vc_ei"):
        mode_dir = model_dir / mode
        if not mode_dir.exists():
            continue
        rs: List[Dict] = []
        for f in mode_dir.glob("*.json"):
            if f.name.startswith("_"):
                continue
            d = _load_json(f)
            if isinstance(d, dict):
                rs.append(d)
        if not rs:
            continue
        non_trivial = [r for r in rs if r.get("num_conflicts", 0) > 0]
        out["per_mode"][mode] = {
            "n": len(rs),
            "n_non_trivial": len(non_trivial),
            "conflict_free_pct_all": 100 * sum(1 for r in rs if r.get("success")) / len(rs),
            "conflict_free_pct_non_trivial": (
                100 * sum(1 for r in non_trivial if r.get("success")) / len(non_trivial)
                if non_trivial else 0.0
            ),
            "mean_pre_strict": 100 * statistics.mean(r.get("pre_strict_direction_match", 0) or 0 for r in rs),
            "mean_post_strict": 100 * statistics.mean(r.get("post_strict_direction_match", 0) or 0 for r in rs),
            "mean_delta_strict": 100 * statistics.mean(r.get("strict_direction_delta", 0) or 0 for r in rs),
        }
    return out


def summarize_exp06() -> Optional[Dict]:
    raw = _load_json(ROOT / "exp06" / "raw.json")
    if not isinstance(raw, list) or not raw:
        return None
    cohort: Dict = {}
    for r in raw:
        cohort.setdefault((r["family"], r["err_type"], r["num_errors"]), []).append(r)
    out: Dict = {"n": len(raw), "per_cohort": {}}
    for (f, e, n_err), rs in cohort.items():
        out["per_cohort"][f"{f}__{e}__{n_err}"] = {
            "n": len(rs),
            "heuristic_cf_pct": 100 * sum(1 for r in rs if r.get("heur_conflict_free")) / len(rs),
            "random_cf_pct": 100 * sum(1 for r in rs if r.get("rand_conflict_free")) / len(rs),
            "heuristic_dir_acc_pct": 100 * statistics.mean(r.get("heur_dir_acc", 0) or 0 for r in rs),
            "random_dir_acc_pct": 100 * statistics.mean(r.get("rand_dir_acc", 0) or 0 for r in rs),
        }
    return out


def summarize_exp07() -> Optional[Dict]:
    files = list((ROOT / "exp07").glob("*.json"))
    rs: List[Dict] = []
    for f in files:
        if f.name == "summary.md":
            continue
        d = _load_json(f)
        if isinstance(d, dict) and "num_conflicts" in d:
            rs.append(d)
    if not rs:
        return None
    non_trivial = [r for r in rs if r.get("num_conflicts", 0) > 0]
    return {
        "n": len(rs),
        "n_non_trivial": len(non_trivial),
        "conflict_free_pct_non_trivial": (
            100 * sum(1 for r in non_trivial if r.get("success")) / len(non_trivial)
            if non_trivial else 0.0
        ),
        "mean_pre_strict": 100 * statistics.mean(r.get("pre_strict_direction_match", 0) or 0 for r in rs),
        "mean_post_strict": 100 * statistics.mean(r.get("post_strict_direction_match", 0) or 0 for r in rs),
        "mean_delta_strict": 100 * statistics.mean(r.get("strict_direction_delta", 0) or 0 for r in rs),
    }


def summarize_exp08() -> Optional[Dict]:
    mango = _load_json(ROOT / "exp08" / "mango.json")
    synth = _load_json(ROOT / "exp08" / "synth.json")
    out: Dict = {}
    if isinstance(mango, list) and mango:
        rs = [r for r in mango if "error" not in r and r.get("num_conflicts", 0) > 0]
        if rs:
            out["mango"] = {
                "n_non_trivial": len(rs),
                "conflict_free_pct": 100 * sum(1 for r in rs if r.get("success")) / len(rs),
                "mean_delta_strict": 100 * statistics.mean(r.get("delta_strict", 0) or 0 for r in rs),
            }
    if isinstance(synth, list) and synth:
        cohort: Dict = {}
        for r in synth:
            cohort.setdefault((r["family"], r["err_type"], r["prefer_remove"]), []).append(r)
        out["synth_compare"] = {
            f"{family}__{err}__{'remove' if prefer else 'rotate'}": {
                "n": len(rs),
                "conflict_free_pct": 100 * sum(1 for r in rs if r.get("conflict_free")) / len(rs),
                "dir_acc_pct": 100 * statistics.mean(r.get("dir_acc", 0) or 0 for r in rs),
            }
            for (family, err, prefer), rs in cohort.items()
        }
    return out


def summarize_exp09() -> Optional[Dict]:
    raw = _load_json(ROOT / "exp09" / "raw.json")
    if not isinstance(raw, list) or not raw:
        return None
    cohort: Dict = {}
    for r in raw:
        if "family" in r:
            cohort.setdefault((r["family"], r["err_type"]), []).append(r)
    out: Dict = {"per_cohort": {}}
    for (f, e), rs in cohort.items():
        out["per_cohort"][f"{f}__{e}"] = {
            "n": len(rs),
            "heuristic_cf_pct": 100 * sum(1 for r in rs if r.get("heuristic_remove_conflict_free")) / len(rs),
            "heuristic_dir_acc_pct": 100 * statistics.mean(r.get("heuristic_remove_dir_acc", 0) or 0 for r in rs),
            "llm_baseline_cf_pct": 100 * sum(1 for r in rs if r.get("llm_baseline_conflict_free")) / len(rs),
            "llm_edge_impact_cf_pct": 100 * sum(1 for r in rs if r.get("llm_edge_impact_conflict_free")) / len(rs),
            "llm_vc_ei_cf_pct": 100 * sum(1 for r in rs if r.get("llm_vc_ei_conflict_free")) / len(rs),
            "llm_vc_ei_dir_acc_pct": 100 * statistics.mean(r.get("llm_vc_ei_dir_acc", 0) or 0 for r in rs),
        }
    return out


def main() -> int:
    summary = {
        "exp01_localization": summarize_exp01(),
        "exp02_scoring": summarize_exp02(),
        "exp03_heuristic_repair": summarize_exp03(),
        "exp04_llm_synth": summarize_exp04(),
        "exp05_mango_llm": summarize_exp05(),
        "exp06_difficulty": summarize_exp06(),
        "exp07_heur_on_mango": summarize_exp07(),
        "exp08_remove_first": summarize_exp08(),
        "exp09_head_to_head": summarize_exp09(),
    }
    (ROOT / "FINAL_REPORT.json").write_text(json.dumps(summary, indent=2))
    md = ["# Final consolidated report\n",
          "(Numbers below are aggregated from each experiment's `raw.json` / per-game files. "
          "Missing entries indicate the experiment has not finished yet.)\n"]
    for key, value in summary.items():
        md.append(f"\n## {key}")
        if value is None:
            md.append("_(no data yet)_")
        else:
            md.append("```json")
            md.append(json.dumps(value, indent=2))
            md.append("```")
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(md) + "\n")
    print("Wrote results/FINAL_REPORT.{md,json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
