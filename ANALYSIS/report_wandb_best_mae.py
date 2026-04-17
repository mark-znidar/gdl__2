"""Aggregate best/test_mae from W&B lhks-pe-2seed; average across seeds; write markdown table."""
import re
from collections import defaultdict

import wandb

ENTITY = "znidar-mark-stanford-university"
PROJECT = "lhks-pe-2seed"
OUT_MD = __file__.replace("report_wandb_best_mae.py", "wandb_best_test_mae_table.md")

# Run name: ZINC-subset.GPS.GINE+Transformer[+<variant>].r<seed>
RUN_RE = re.compile(
    r"^ZINC-subset\.GPS\.GINE\+Transformer(?:\+(.+))?\.r(\d+)$"
)


def parse_run(name: str):
    m = RUN_RE.match(name)
    if not m:
        return None, None
    variant, seed = m.group(1), int(m.group(2))
    if variant is None:
        variant = "no PE"
    return variant, seed


def main():
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    by_variant: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for r in runs:
        v, seed = parse_run(r.name or "")
        if v is None:
            continue
        mae = r.summary.get("best/test_mae") if hasattr(r.summary, "get") else None
        if mae is None:
            continue
        by_variant[v].append((seed, float(mae)))

    rows = []
    for variant, pairs in sorted(by_variant.items()):
        pairs.sort(key=lambda x: x[0])
        maes = [m for _, m in pairs]
        mean_mae = sum(maes) / len(maes)
        seeds_str = ",".join(str(s) for s, _ in pairs)
        rows.append(
            {
                "model": variant,
                "n_seeds": len(maes),
                "seeds": seeds_str,
                "per_seed_mae": ", ".join(f"{m:.5f}" for m in maes),
                "mean_best_test_mae": mean_mae,
            }
        )

    rows.sort(key=lambda r: r["mean_best_test_mae"])

    lines = [
        "# Best test MAE by model (`best/test_mae` from W&B)",
        "",
        f"**Project:** `{ENTITY}/{PROJECT}`  ",
        "Lower is better. When multiple `.r*` runs exist for the same variant, **mean** is reported.",
        "",
        "| Model (variant) | Seeds | Per-seed MAE | **Mean best test MAE** |",
        "|---|---:|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['n_seeds']} | {r['per_seed_mae']} | **{r['mean_best_test_mae']:.5f}** |"
        )
    lines.append("")

    md = "\n".join(lines)
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(md)
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
