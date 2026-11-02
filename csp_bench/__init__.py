"""csp_bench — diagnostic metrics for generative crystal structure prediction.

Modules map directly onto the axes in "Diagnostic Benchmarking of Deep
Generative CSP — Protocol":

    symmetry.py   Axis A — symmetry ladder (system / SG / Wyckoff / decorated)
    lattice.py    Axis B — lattice and unit-cell error
    bonding.py    Axis C — coordination, distortion, topology
    energy.py     Axis D — energy, phonons, soft-mode overlap
    matcher.py    frozen StructureMatcher used by every axis
    cascade.py    post-processing cascade + F0-F9 failure taxonomy
    topk.py       top-1 / top-K aggregation, pass@k, ranker sensitivity
"""

from . import symmetry, lattice, bonding, energy, matcher, cascade, topk

__all__ = [
    "symmetry",
    "lattice",
    "bonding",
    "energy",
    "matcher",
    "cascade",
    "topk",
]
