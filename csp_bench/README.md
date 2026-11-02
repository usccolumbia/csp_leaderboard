# csp_bench

The diagnostic package of the paper. It explains *why* a predicted structure fails to match its reference, along four
axes and a ten-bucket failure-mode cascade. The match metrics of the leaderboard (top-k, space group, statistics) are in
[`../metrics`](../metrics).

| Module | Contents |
|---|---|
| `symmetry.py` | Axis A: space-group ladder with spglib over increasing `symprec` (crystal system, space group, Wyckoff, decorated) |
| `lattice.py` | Axis B: lattice error from the polar decomposition of the deformation gradient (volume and shape), cubic deviation |
| `bonding.py` | Axis C: CrystalNN coordination after site alignment, polyhedral distortion, graph-hash topology |
| `energy.py` | Axis D: relaxed energy against the reference, phonons, soft-mode overlap |
| `mlip.py` | interface to the interatomic potential used by Axis D and the cascade (MatterSim, CHGNet, MACE, UMA) |
| `cascade.py` | the F0–F9 failure-mode cascade |
| `topk.py` | top-1 / top-k aggregation, pass@k estimate, ranker sensitivity |
| `matcher.py` | the framework's loose matcher (`ltol` 0.3, `stol` 0.5, `angle_tol` 10) |
| `structure_utils.py` | primitive and standard cells, deformation gradient, volume rescaling |

## Failure-mode buckets

The stages are applied in order to the raw (unrelaxed) top-1 candidate; the first stage at which it matches the
reference gives its bucket.

| Bucket | Meaning |
|---|---|
| F0 | matches as generated |
| F1 | matches after loose symmetrization (spglib, `symprec` 0.5) |
| F2 | matches after rescaling to the reference volume |
| F3a | matches after a symmetry-constrained relaxation |
| F3b | matches after a free relaxation |
| F8 | relaxes to a structure lower in energy than the reference |
| F4 | correct decorated Wyckoff positions, wrong energy basin |
| F5 | correct space group, wrong Wyckoff decoration |
| F6 | correct crystal system or topology, wrong space group |
| F7 | correct local coordination, wrong assembly |
| F9 | wrong local coordination |

## Installation and use

From the repository root:

```bash
pip install -e .                  # core dependencies
pip install -e ".[mattersim]"     # the potential used in the paper
pytest csp_bench/tests            # 22 tests; tests that need a potential are marked slow and skipped
```

```python
from pymatgen.core import Structure
from csp_bench.cascade import assign_bucket
from csp_bench.mlip import get_mlip
from metrics.csp_metrics import MATCHER          # the leaderboard matcher (ltol 0.2, stol 0.3, angle_tol 5)

mlip = get_mlip("mattersim", model="MatterSim-v1.0.0-5M", device="cuda")
reference = Structure.from_file("testsets/T180/cifs/mp-788.cif").get_primitive_structure(tolerance=0.01)
candidate = Structure.from_file("my_method/T180/mp-788/rank1.cif")    # raw, unrelaxed top-1 candidate
trace = assign_bucket(candidate, reference, mlip, matcher=MATCHER)
print(trace.bucket)
```

In the paper the cascade uses the leaderboard matcher, so that the F0 share equals the raw top-1 match rate, and the
loose matcher of `matcher.py` (the default when no matcher is passed) as a sensitivity analysis. Relaxations use
MatterSim v1.0.0-5M (BFGS with a Fréchet cell filter, fmax 0.02 eV/Å, at most 500 steps), distinct from the ORB-v3
potential that relaxes and ranks the exported candidates.
