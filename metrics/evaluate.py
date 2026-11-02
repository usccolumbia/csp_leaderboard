"""Score one method on one test set.

Predictions are read from <predictions>/<material_id>/*.cif; the files of a target are ranked by natural filename
order (rank1.cif, rank2.cif, ..., rank10.cif), and only the first 20 are scored.

    python metrics/evaluate.py --testset testsets/T180 --predictions my_method/T180 --out results_my_method

writes <out>/per_target.csv and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

from pymatgen.core import Structure

from csp_metrics import natural_key, score_target, space_group, summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", required=True, type=Path, help="test-set directory, e.g. testsets/T180")
    ap.add_argument("--predictions", required=True, type=Path, help="directory with one subdirectory of CIFs per target")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    queries = json.loads((args.testset / "queries.json").read_text())
    rows = []
    for q in queries:
        mid = q["material_id"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = Structure.from_file(args.testset / "cifs" / f"{mid}.cif")
        files = sorted((args.predictions / mid).glob("*.cif"), key=natural_key)
        r = dict(material_id=mid, query_formula=q["formula"])  # formula of the primitive cell, e.g. Co2Te2
        r.update(score_target(ref, files, q["formula"]))
        r["ref_sg"], r["ref_sg_symprec001"] = space_group(ref, 0.1), space_group(ref, 0.01)
        rows.append(r)
        print(mid, r["first_match_rank"], flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cols = ["material_id", "query_formula", "n_candidates", "first_match_rank", "sg_top1", "ref_sg",
            "sg_top1_symprec001", "ref_sg_symprec001", "errors"]
    with open(args.out / "per_target.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**{c: r[c] for c in cols}, "errors": " | ".join(r["errors"])})
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
