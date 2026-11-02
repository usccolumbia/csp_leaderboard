"""Metrics of the CSP leaderboard.

Scoring of one target
    A method returns up to 20 candidate structures for a target, in rank order. A candidate is valid if its element
    counts equal those of the query formula (the primitive cell of the target). A candidate matches the reference if
    pymatgen's StructureMatcher (ltol 0.2, stol 0.3, angle_tol 5, primitive_cell, scale, no supercell) finds a fit.
    The score of the target is the rank of the first matching candidate (None if no candidate matches). Ranks are
    positions in the submitted list, so an invalid candidate still uses up its rank.

Aggregation over a test set
    top-k     number of targets whose first match has rank <= k (targets without candidates count as failures)
    SG top-1  number of targets whose rank-1 candidate has the space group of the reference, both determined with
              spglib at the same symprec (0.1 by default; 0.01 reported as a sensitivity check)
    consensus number of targets whose rank-1 candidate matches the reference and has its space group (symprec 0.1)

Statistics
    bootstrap_ci  percentile bootstrap over targets, 10,000 resamples, seed 2026, the same resampled targets for every
                  method
    mcnemar_exact two-sided exact McNemar test on paired per-target outcomes
    holm          Holm step-down correction over a family of comparisons
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True, attempt_supercell=False)
TOP_K = 20
KS = (1, 5, 20)


def natural_key(path: Path) -> list:
    """Sort key that orders rank2.cif before rank10.cif."""
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", Path(path).name)]


def space_group(structure: Structure, symprec: float = 0.1) -> int | None:
    try:
        return SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_number()
    except Exception:
        return None


def score_target(reference: Structure, candidates: list[Path], query_formula: str, top_k: int = TOP_K) -> dict:
    """Score the ranked candidate files of one target against its reference structure."""
    query = Composition(query_formula)
    out = dict(n_candidates=0, first_match_rank=None, sg_top1=None, sg_top1_symprec001=None, errors=[])
    for rank, path in enumerate(candidates[:top_k], 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = Structure.from_file(path)
            if not s.composition.almost_equals(query):
                raise ValueError(f"element counts {s.composition.formula} differ from the query {query_formula}")
            out["n_candidates"] += 1
            if rank == 1:
                out["sg_top1"] = space_group(s, 0.1)
                out["sg_top1_symprec001"] = space_group(s, 0.01)
            if out["first_match_rank"] is None and MATCHER.fit(reference, s):
                out["first_match_rank"] = rank
        except Exception as e:
            out["errors"].append(f"{Path(path).name}: {e}")
    return out


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-target rows (from score_target plus the reference space groups) over a test set."""
    n = len(rows)
    hit = lambda r, k: r["first_match_rank"] is not None and r["first_match_rank"] <= k
    s = dict(n_targets=n, targets_with_candidates=sum(r["n_candidates"] > 0 for r in rows))
    for k in KS:
        s[f"top{k}"] = sum(hit(r, k) for r in rows)
        s[f"top{k}_pct"] = round(100 * s[f"top{k}"] / n, 1)
    sg = lambda r, p: r[f"sg_top1{p}"] is not None and r[f"sg_top1{p}"] == r[f"ref_sg{p}"]
    s["sg_top1"] = sum(sg(r, "") for r in rows)
    s["sg_top1_symprec001"] = sum(sg(r, "_symprec001") for r in rows)
    s["consensus_top1"] = sum(hit(r, 1) and sg(r, "") for r in rows)
    return s


def bootstrap_ci(hits: dict[str, np.ndarray], n_resamples: int = 10_000, seed: int = 2026, level: float = 95) -> dict:
    """Percentile bootstrap interval (in %) of the match rate of each method.

    hits maps a method name to a 0/1 array over the same targets in the same order; every method is evaluated on the
    same resampled target indices."""
    n = len(next(iter(hits.values())))
    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    tail = (100 - level) / 2
    out = {}
    for name, h in hits.items():
        h = np.asarray(h, dtype=float)
        lo, hi = np.percentile(h[idx].mean(axis=1) * 100, [tail, 100 - tail])
        out[name] = (float(lo), float(hi))
    return out


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact McNemar test of paired 0/1 outcomes.

    Returns (targets solved by x only, targets solved by y only, p-value)."""
    x, y = np.asarray(x, dtype=bool), np.asarray(y, dtype=bool)
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return b, c, min(1.0, p)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values for one family of comparisons."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out
