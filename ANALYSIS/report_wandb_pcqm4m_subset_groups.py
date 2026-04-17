"""Report best/test_mae (mean ± std) per group for PCQM4Mv2-subset experiments.

Pulls runs from two W&B projects and groups them:
  - pcqm4m-subset-lhks-mlp-ablation:
      * LHKS-fixed        -> names contain "+LHKS-fixed"
      * LHKS (MLP3)       -> names contain "+LHKS." and "MLP3" (excludes MLP2/MLP4)
  - pcqm4m-subset-stability-baselines:
      * LapPE             -> names contain "+LapPE"
      * SignNetMLP        -> names contain "+SignNetMLP"   (excludes SignNetDS)

Metric: best/test_mae. Runs whose name contains "CRASHED" are skipped.
Writes markdown table next to this script.
"""
from __future__ import annotations

import math
from collections import defaultdict

import wandb

ENTITY = "znidar-mark-stanford-university"
ABLATION_PROJECT = "pcqm4m-subset-lhks-mlp-ablation"
BASELINES_PROJECT = "pcqm4m-subset-stability-baselines"
OUT_MD = __file__.replace("report_wandb_pcqm4m_subset_groups.py",
                          "wandb_pcqm4m_subset_groups_table.md")

METRIC_KEY = "best/test_mae"


def classify(name: str, project: str):
    """Return a group label for the run, or None to skip."""
    if not name or "CRASHED" in name.upper():
        return None

    if project == ABLATION_PROJECT:
        if "+LHKS-fixed" in name:
            return "LHKS-fixed"
        if "+LHKS." in name and "MLP3" in name and "MLP3" in name.split("+LHKS.")[-1]:
            # Ensure we don't catch MLP2/MLP4 names that happen to contain "3" elsewhere.
            tail = name.split("+LHKS.")[-1]
            if "MLP3" in tail and "MLP2" not in tail and "MLP4" not in tail:
                return "LHKS (MLP3)"
        return None

    if project == BASELINES_PROJECT:
        if "+LapPE" in name:
            return "LapPE"
        if "+SignNetMLP" in name:
            return "SignNetMLP"
        return None

    return None


def fetch_metric(run):
    """Return float best/test_mae or None."""
    try:
        val = run.summary.get(METRIC_KEY) if hasattr(run.summary, "get") else None
    except Exception:
        val = None
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample std
        std = math.sqrt(var)
    else:
        std = float("nan")
    return mean, std


def main():
    api = wandb.Api(timeout=120)

    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for project in (ABLATION_PROJECT, BASELINES_PROJECT):
        print(f"Fetching {ENTITY}/{project} ...")
        runs = list(api.runs(f"{ENTITY}/{project}"))
        for r in runs:
            name = r.name or ""
            label = classify(name, project)
            if label is None:
                continue
            mae = fetch_metric(r)
            if mae is None:
                print(f"  [skip no metric] {name}")
                continue
            groups[label].append((name, mae))
            print(f"  [{label}] {name}: {mae:.5f}")

    # Build markdown table sorted by mean.
    rows = []
    for label, items in groups.items():
        items.sort()
        maes = [m for _, m in items]
        mean, std = stats(maes)
        rows.append(
            {
                "group": label,
                "n": len(maes),
                "runs": "<br>".join(n for n, _ in items),
                "per_run": ", ".join(f"{m:.5f}" for m in maes),
                "mean": mean,
                "std": std,
            }
        )
    rows.sort(key=lambda r: r["mean"])

    lines = [
        "# PCQM4Mv2-subset — best/test_mae by group (mean ± std)",
        "",
        f"**Entity:** `{ENTITY}`  ",
        f"**Projects:** `{ABLATION_PROJECT}`, `{BASELINES_PROJECT}`  ",
        f"**Metric:** `{METRIC_KEY}` (lower is better). Std is sample std (ddof=1).",
        "",
        "| Group | n | Per-seed MAE | **Mean ± Std** |",
        "|---|---:|---|---:|",
    ]
    for r in rows:
        std_str = "—" if math.isnan(r["std"]) else f"{r['std']:.5f}"
        lines.append(
            f"| {r['group']} | {r['n']} | {r['per_run']} | **{r['mean']:.5f} ± {std_str}** |"
        )
    lines.append("")
    lines.append("## Run names per group")
    for r in rows:
        lines.append("")
        lines.append(f"**{r['group']}** (n={r['n']}):")
        for name, mae in sorted(groups[r["group"]]):
            lines.append(f"- `{name}` → {mae:.5f}")

    md = "\n".join(lines) + "\n"
    with open(OUT_MD, "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
