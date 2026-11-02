# CSP leaderboard

**Live leaderboard:** [MaterialsAtlas.org/csp-leaderboard](http://MaterialsAtlas.org/csp-leaderboard)

Test sets and key results for crystal structure prediction (CSP) from the paper **"A diagnostic framework and leaderboard
for benchmarking crystal structure prediction"** (Y. Feng, L. Tao, J. Hu, R. Dong, L. Wei, S. Dey, R. Rogers, M. Hu and
J. Hu; in preparation). Machine Learning and Evolution Lab, University of South Carolina.

* **Four leakage-audited test sets** (release v3.1, 8 October 2026): T46, T180, T240 and T800, with reference CIFs and
  the formula queries given to every method.
* **The MP-20 audit and replacement records** for these sets.
* **Key results**: top-k match rates with bootstrap intervals and paired tests on every set, and per-target scores and
  failure modes of every method on T180.

## Test sets

| Set | Targets | Elements | Source | Use |
|---|---:|---|---|---|
| T46 | 46 | 2–4 | Materials Project (33 targets shared with T180) | small set; also run with global structure searches |
| **T180** | 180 | 2–4 | Materials Project, 60 binary / 60 ternary / 60 quaternary | main test set |
| T240 | 240 | 2–5 | T180 + 60 five-element targets | larger compositions |
| T800 | 800 | 1–6 | MPTS-52 time-split test set; 686 cells above 20 atoms | larger cells; models trained on MPTS-52 |

Each set directory in [`testsets/`](testsets) contains `cifs/<material_id>.cif` (reference structures),
`queries.csv` / `queries.json` (the input given to every method), `metadata.csv` and `records.json` (reference metadata),
`template_exclusions.json` (compositions a template method must exclude) and `manifest.json`.
`testsets/CIF_SHA256SUMS.json` lists the SHA-256 of every CIF.

**Task.** A method receives only the formula of the target's primitive cell with integer counts (for example `Co2Te2`
for CoTe), with no lattice, coordinates or symmetry, and returns up to 20 ranked candidate structures.

**Leakage control.** Every target was compared with all 36,183 MP-20 training and validation structures with pymatgen
`StructureMatcher` at two tolerance settings, with and without supercell matching, in both directions. Matching targets
were replaced (T46: 7, T180: 36, T240: 41, T800: 34, including one replacement for reference quality); no target of the
released sets matches an MP-20 training or validation structure, and every set has unique compositions. T46, T180 and
T240 are audited against MP-20 only; their replacements come from MPTS-52 records, so they are not guaranteed free of
MPTS-52 training data. T800 replacements come from the MPTS-52 test split only. See [`audit/`](audit).

**Reference quality.** Every reference has a DFT energy above the convex hull of at most 0.1 eV/atom. All T180
references are phonon-stable (162) or experimentally confirmed (18).

## Evaluation

A candidate matches the reference if pymatgen `StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True,
scale=True, attempt_supercell=False)` returns a fit. Top-k counts a target as solved if any of the first k candidates
matches. Every method's candidates are relaxed with the ORB-v3 machine-learned interatomic potential and ranked by
energy. Targets without a valid candidate count as failures. The CSV files in `results/` also give 95% bootstrap
intervals over targets (10,000 resamples); `p_holm` is an exact McNemar test of top-1 against TCSPv3, Holm-corrected within each set.

The metric code is in [`metrics/`](metrics): `evaluate.py` scores a method on a test set and `compare.py` computes the
intervals and paired tests.

## Results on T180

Targets matched out of 180 (rate in %).

| # | Method | Trained on | Top-1 | Top-5 | Top-20 | p<sub>Holm</sub> vs TCSPv3 |
|---:|---|---|---:|---:|---:|---:|
| 1 | TCSPv3 | template library | 130 (72.2) | 159 | 163 (90.6) |  |
| 2 | CrystalFlow | MP-20 | 115 (63.9) | 121 | 130 (72.2) | 0.036 |
| 3 | MaskGXT † | MP-20 | 113 (62.8) | 116 | 119 (66.1) | 0.027 |
| 4 | Crystalite | MP-20 | 112 (62.2) | 120 | 132 (73.3) | 0.035 |
| 5 | OMatG | MP-20 | 107 (59.4) | 116 | 124 (68.9) | 0.003 |
| 6 | Uni-3DAR | MP-20 | 105 (58.3) | 110 | 114 (63.3) | 0.003 |
| 7 | EquiCSP | MP-20 | 102 (56.7) | 111 | 119 (66.1) | <0.001 |
| 8 | uFlowCSP | MP-20 | 101 (56.1) | 117 | 120 (66.7) | <0.001 |
| 9 | MatterGen | MP-20 | 97 (53.9) | 110 | 119 (66.1) | <0.001 |
| 10 | DiffCSP | MP-20 | 97 (53.9) | 108 | 115 (63.9) | <0.001 |
| 11 | cond-CDVAE | MP-20 | 72 (40.0) | 82 | 86 (47.8) | <0.001 |
| 12 | GemsDiff | MP-20 | 62 (34.4) | 64 | 66 (36.7) | <0.001 |
| 13 | TGDMat | MP-20 | 50 (27.8) | 52 | 53 (29.4) | <0.001 |
| 14 | CDVAE | MP-20 | 48 (26.7) | 52 | 53 (29.4) | <0.001 |
| 15 | SymmCD | MP-20 | 40 (22.2) | 45 | 49 (27.2) | <0.001 |
| 16 | CrystalGRW | Alex-MP-20 | 23 (12.8) | 24 | 24 (13.3) | <0.001 |
| 17 | CrystalFormer | Alex-20s | 12 (6.7) | 13 | 13 (7.2) | <0.001 |

† MaskGXT: locally retrained MP-20 checkpoint; supports cells of up to 20 atoms (4 T180 targets count as misses).
On T180 the Holm correction is over the 15 other generators, as in the paper; the MaskGXT p-value is unadjusted.

## Top-1 on all test sets

Targets matched at top-1 (rate in %); – means the method was not run on that set. Rows follow the T180 order; the
global structure searches were run on T46 only.

| Method | T180 (n=180) | T46 (n=46) | T240 (n=240) | T800 (n=800) |
|---|---:|---:|---:|---:|
| TCSPv3 | 130 (72.2) | 35 (76.1) | 149 (62.1) | 518 (64.8) |
| CrystalFlow | 115 (63.9) | 26 (56.5) | 130 (54.2) | 70 (8.8) |
| MaskGXT | 113 (62.8) | 30 (65.2) | 139 (57.9) | 76 (9.5) |
| Crystalite | 112 (62.2) | 27 (58.7) | 124 (51.7) | 72 (9.0) |
| OMatG | 107 (59.4) | 27 (58.7) | 117 (48.8) | 73 (9.1) |
| Uni-3DAR | 105 (58.3) | 32 (69.6) | 124 (51.7) | 64 (8.0) |
| EquiCSP | 102 (56.7) | 27 (58.7) | 112 (46.7) | 65 (8.1) |
| uFlowCSP | 101 (56.1) | 27 (58.7) | 104 (43.3) | 58 (7.2) |
| MatterGen | 97 (53.9) | 25 (54.3) | 103 (42.9) | 43 (5.4) |
| DiffCSP | 97 (53.9) | 26 (56.5) | 105 (43.8) | 61 (7.6) |
| cond-CDVAE | 72 (40.0) | 17 (37.0) | 76 (31.7) | 37 (4.6) |
| GemsDiff | 62 (34.4) | 15 (32.6) | 65 (27.1) | 31 (3.9) |
| TGDMat | 50 (27.8) | 17 (37.0) | 50 (20.8) | 21 (2.6) |
| CDVAE | 48 (26.7) | 12 (26.1) | 48 (20.0) | 18 (2.2) |
| SymmCD | 40 (22.2) | 7 (15.2) | 42 (17.5) | 9 (1.1) |
| CrystalGRW | 23 (12.8) | 8 (17.4) | 23 (9.6) | 7 (0.9) |
| CrystalFormer | 12 (6.7) | 3 (6.5) | 14 (5.8) | – |
| USPEX ‡ | – | 26 (56.5) | – | – |
| MAGUS | – | 23 (50.0) | – | – |
| XtalOpt | – | 17 (37.0) | – | – |

‡ USPEX is being re-run so that every search relaxes candidates with MatterSim; its T46 numbers are provisional.

Full tables are in [`results/`](results): `<set>_summary.csv` (top-1/5/20 with intervals, space-group agreement,
consensus and paired tests) and `T180_per_target.csv` (for every target and method: matched at top-1, matched at
top-20, and the failure-mode bucket F0–F9 of the raw top-1 candidate).

## Submitting a method

Run your method on the queries of a test set, export up to 20 ranked candidates per target, and open an issue or pull
request with the candidates and a short description (training data, checkpoint, sampling settings). We evaluate them
with the protocol above and add the results.

## License

MIT (see [`LICENSE`](LICENSE)). Reference structures are derived from the Materials Project (CC BY 4.0) and MPTS-52.

