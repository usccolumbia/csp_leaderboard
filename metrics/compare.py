"""Bootstrap intervals and paired tests for several methods on one test set.

Input is either the per_target.csv files written by evaluate.py, one per method:

    python metrics/compare.py --per-target TCSPv3=out_tcspv3/per_target.csv CrystalFlow=out_cf/per_target.csv \
        --reference TCSPv3

or a wide table with a 0/1 column <method>_top<k> per method, such as results/T180_per_target.csv:

    python metrics/compare.py --wide results/T180_per_target.csv --reference TCSPv3 --exclude MaskGXT

For every method and k in (1, 5, 20) the output gives the targets matched, the rate, and its 95% percentile bootstrap
interval. For top-1 it also gives the exact McNemar test against the reference method, Holm-corrected over all
compared methods except those listed with --exclude (their p-values are reported unadjusted).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from csp_metrics import KS, bootstrap_ci, holm, mcnemar_exact


def load_per_target(specs: list[str]) -> tuple[list[str], dict[str, dict[int, np.ndarray]]]:
    tables = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        tables[name] = {r["material_id"]: r["first_match_rank"] for r in csv.DictReader(open(path))}
    ids = sorted(next(iter(tables.values())))
    assert all(sorted(t) == ids for t in tables.values()), "all methods must cover the same targets"
    hits = {name: {k: np.array([t[m] != "" and int(t[m]) <= k for m in ids]) for k in KS} for name, t in tables.items()}
    return ids, hits


def load_wide(path: Path) -> tuple[list[str], dict[str, dict[int, np.ndarray]]]:
    rows = sorted(csv.DictReader(open(path)), key=lambda r: r["material_id"])
    names = [c[: -len("_top1")] for c in rows[0] if c.endswith("_top1")]
    ks = [k for k in KS if f"{names[0]}_top{k}" in rows[0]]
    hits = {n: {k: np.array([r[f"{n}_top{k}"] == "1" for r in rows]) for k in ks} for n in names}
    return [r["material_id"] for r in rows], hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--per-target", nargs="+", metavar="NAME=CSV")
    src.add_argument("--wide", type=Path)
    ap.add_argument("--reference", required=True, help="method that every other method is tested against")
    ap.add_argument("--exclude", nargs="*", default=[], help="methods left out of the Holm family")
    ap.add_argument("--out", type=Path, help="optional CSV output")
    args = ap.parse_args()

    ids, hits = load_wide(args.wide) if args.wide else load_per_target(args.per_target)
    n = len(ids)
    ks = sorted(next(iter(hits.values())))
    ci = {k: bootstrap_ci({m: h[k] for m, h in hits.items()}) for k in ks}

    others = [m for m in hits if m != args.reference]
    tests = {m: mcnemar_exact(hits[args.reference][1], hits[m][1]) for m in others}
    adjusted = holm({m: tests[m][2] for m in others if m not in args.exclude})

    rows = []
    for m in sorted(hits, key=lambda m: -int(hits[m][1].sum())):
        r = dict(method=m)
        for k in ks:
            h = int(hits[m][k].sum())
            r[f"top{k}"], r[f"top{k}_pct"] = h, round(100 * h / n, 1)
            r[f"top{k}_ci95"] = f"{ci[k][m][0]:.1f}-{ci[k][m][1]:.1f}"
        if m in tests:
            only_ref, only_m, p = tests[m]
            r.update(only_reference=only_ref, only_method=only_m, p=f"{p:.3g}",
                     p_holm=f"{adjusted[m]:.3g}" if m in adjusted else "unadjusted")
        rows.append(r)

    cols = list(dict.fromkeys(c for r in rows for c in r))
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r.get(c, "")) for c in cols))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
