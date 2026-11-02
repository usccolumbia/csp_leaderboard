# Metrics

The code that scores CSP predictions on the test sets, as used for the leaderboard and the paper.
Requires Python 3.9+, `pymatgen` and `numpy`.

| File | Contents |
|---|---|
| `csp_metrics.py` | the matcher, per-target scoring, top-k / space-group / consensus counts, bootstrap intervals, exact McNemar test, Holm correction |
| `evaluate.py` | scores one method on one test set and writes `per_target.csv` and `summary.json` |
| `compare.py` | bootstrap intervals and paired tests for several methods on one test set |

## Scoring one method

Put up to 20 ranked candidates per target in one directory per target, named so that natural filename order is the
rank order:

```
my_method/T180/mp-24658/rank1.cif
my_method/T180/mp-24658/rank2.cif
...
```

Then run, from the repository root:

```bash
python metrics/evaluate.py --testset testsets/T180 --predictions my_method/T180 --out results_my_method
```

## Definitions

* **Valid candidate.** Its element counts equal those of the query formula, which is the target's primitive cell (for
  example `Co2Te2`, not `CoTe`). Invalid candidates still use up their rank. Some reference CIFs are stored as
  conventional cells; the matcher reduces both structures to primitive cells.
* **Match.** pymatgen `StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True,
  attempt_supercell=False)` finds a fit between the reference and the candidate.
* **Top-k.** The target counts as solved if the first matching candidate has rank ≤ k (k = 1, 5, 20). Targets without
  candidates count as failures, so the denominator is always the full test set.
* **Space-group agreement (top-1).** The rank-1 candidate has the space group of the reference, both determined with
  spglib at symprec 0.1 (0.01 is also reported).
* **Consensus (top-1).** The rank-1 candidate matches the reference and has its space group.
* **Intervals.** 95% percentile bootstrap over targets, 10,000 resamples, seed 2026, with the same resampled targets
  for every method.
* **Paired test.** Two-sided exact McNemar test of top-1 outcomes against a reference method, Holm-corrected over the
  compared methods.

On the leaderboard every method's candidates are first relaxed with ORB-v3 (500 steps, fmax 0.01 eV/Å) and ranked by
energy; the metrics are then applied to that ranked list.

## Reproducing the published T180 statistics

```bash
python metrics/compare.py --wide results/T180_per_target.csv --reference TCSPv3 --exclude MaskGXT
```

This gives the top-1 and top-20 intervals and the Holm-corrected p-values of the paper (for example CrystalFlow:
30 targets solved by TCSPv3 only, 15 by CrystalFlow only, p<sub>Holm</sub> = 0.036).
