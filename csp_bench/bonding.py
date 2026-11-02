"""Axis C — coordination, local geometry, distortion, topology.

Computed twice by the caller (raw, and after ``rescale_to_volume`` to the
ground-truth volume) — the difference between the two runs attributes a
bonding failure to lattice error rather than to the generator. See
``cascade.bonding_metrics_both`` for that pairing.

Robustness ladder, weakest to strongest claim: coordination number (CN) is
reliable; ChemEnv's continuous symmetry measure (CSM) is noisier; full
periodic-net identification is the most brittle. Report all three, read
them as a ladder, never collapse to one number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure
from scipy.optimize import linear_sum_assignment

_CNN = CrystalNN()


def crystal_nn_graph(structure: Structure, cnn: CrystalNN | None = None):
    cnn = cnn or _CNN
    return cnn.get_bonded_structure(structure)


def site_coordination_numbers(structure: Structure, cnn: CrystalNN | None = None) -> list[int]:
    cnn = cnn or _CNN
    return [cnn.get_cn(structure, i) for i in range(len(structure))]


def align_sites(pred: Structure, gt: Structure) -> list[tuple[int, int]]:
    """Best index correspondence pred[i] <-> gt[j], species-constrained,
    Hungarian assignment on periodic (minimum-image) distance.

    Falls back gracefully when the two structures don't have matching
    composition (returns as many pairs as the smaller species pool allows);
    callers computing CN/CSM accuracy should treat an empty return as
    "not alignable" rather than crash.
    """
    pairs: list[tuple[int, int]] = []
    pred_by_sp: dict[str, list[int]] = {}
    gt_by_sp: dict[str, list[int]] = {}
    for i, site in enumerate(pred):
        pred_by_sp.setdefault(str(site.specie), []).append(i)
    for j, site in enumerate(gt):
        gt_by_sp.setdefault(str(site.specie), []).append(j)

    for sp, p_idx in pred_by_sp.items():
        g_idx = gt_by_sp.get(sp)
        if not g_idx:
            continue
        cost = np.zeros((len(p_idx), len(g_idx)))
        for a, i in enumerate(p_idx):
            for b, j in enumerate(g_idx):
                d = pred.lattice.get_distance_and_image(pred.frac_coords[i], gt.frac_coords[j])[0]
                cost[a, b] = d
        rows, cols = linear_sum_assignment(cost)
        pairs.extend((p_idx[a], g_idx[b]) for a, b in zip(rows, cols))
    return pairs


def _quadratic_elongation(structure: Structure, site_idx: int, cnn: CrystalNN) -> float | None:
    """Robinson et al. quadratic elongation of the coordination polyhedron
    around ``site_idx``: mean squared ratio of bond length to the bond
    length of a "regular" polyhedron of the same volume. Values near 1.0 =
    undistorted; this is the signature that free relaxation typically fixes.
    """
    nbrs = cnn.get_nn_info(structure, site_idx)
    if len(nbrs) < 3:
        return None
    dists = np.array([n["weight"] if "weight" in n else structure[site_idx].distance(n["site"]) for n in nbrs])
    d0 = dists.mean()
    return float(np.mean((dists / d0) ** 2))


def _bond_length_variance(structure: Structure, site_idx: int, cnn: CrystalNN) -> float | None:
    nbrs = cnn.get_nn_info(structure, site_idx)
    if len(nbrs) < 2:
        return None
    dists = np.array([structure[site_idx].distance(n["site"]) for n in nbrs])
    return float(np.var(dists) / np.mean(dists) ** 2)


def _effective_coordination(structure: Structure, site_idx: int, cnn: CrystalNN) -> float | None:
    """ECoN (Hoppe): distance-weighted effective coordination number, robust
    to the hard cutoff CN can be sensitive to near the neighbour-shell edge.
    """
    nbrs = cnn.get_nn_info(structure, site_idx)
    if not nbrs:
        return None
    dists = np.array([structure[site_idx].distance(n["site"]) for n in nbrs])
    d_min = dists.min()
    weights = np.exp(-((dists / d_min) ** 6 - 1))
    d_av = np.sum(dists * weights) / np.sum(weights)
    weights2 = np.exp(1 - (dists / d_av) ** 6)
    return float(np.sum(weights2))


@dataclass
class DistortionMetrics:
    quad_elong: dict[int, float | None]
    econ: dict[int, float | None]
    bl_var: dict[int, float | None]


def distortion_metrics(structure: Structure, cnn: CrystalNN | None = None) -> DistortionMetrics:
    cnn = cnn or _CNN
    n = len(structure)
    return DistortionMetrics(
        quad_elong={i: _quadratic_elongation(structure, i, cnn) for i in range(n)},
        econ={i: _effective_coordination(structure, i, cnn) for i in range(n)},
        bl_var={i: _bond_length_variance(structure, i, cnn) for i in range(n)},
    )


def chemenv_csm(structure: Structure, site_idx: int) -> float | None:
    """Continuous symmetry measure of the coordination environment (0 =
    ideal polyhedron). Lazily imports pymatgen's ChemEnv, which is slow to
    set up per-structure — call sparingly (this axis is explicitly flagged
    "noisier" in the protocol; don't let it gate a whole run).
    """
    try:
        from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
            SimplestChemenvStrategy,
        )
        from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
            LocalGeometryFinder,
        )
    except ImportError:
        return None

    lgf = LocalGeometryFinder()
    lgf.setup_structure(structure=structure)
    se = lgf.compute_structure_environments(only_indices=[site_idx], maximum_distance_factor=1.5)
    strategy = SimplestChemenvStrategy()
    strategy.set_structure_environments(se)
    try:
        ces = strategy.get_site_coordination_environments(structure[site_idx])
    except Exception:
        return None
    csm_values = [c[1] for c in ces if c[1] is not None]
    if not csm_values:
        return None
    # ces: list of (ce_symbol, csm); take the best (lowest CSM) match
    return float(min(csm_values))


def topology_fingerprint(structure: Structure, cnn: CrystalNN | None = None) -> str:
    """Cheap fallback topology invariant: a degree-sequence-and-species
    Weisfeiler-Lehman hash of the CrystalNN bonded-structure graph, refined
    over a few periodic images so it is not just "how many bonds per site".

    This is the protocol's stated fallback, *not* an equivalent of a true
    periodic quotient-graph invariant (RCSR symbol via CrystalNets.jl) — it
    will conflate distinct nets that happen to share local WL colours. Swap
    in ``topology_fingerprint_crystalnets`` when that binding is available.
    """
    import networkx as nx

    bs = crystal_nn_graph(structure, cnn)
    g = bs.graph  # networkx MultiDiGraph with periodic image data on edges
    labels = {i: str(structure[i].specie) for i in g.nodes}
    h = nx.weisfeiler_lehman_graph_hash(nx.Graph(g), node_attr=None, iterations=3)
    # WL hash alone is species-blind; fold in a sorted species/degree summary
    deg_by_sp: dict[str, list[int]] = {}
    for i in g.nodes:
        deg_by_sp.setdefault(labels[i], []).append(g.degree(i))
    summary = tuple(sorted((sp, tuple(sorted(d))) for sp, d in deg_by_sp.items()))
    return f"{h}:{hash(summary)}"


def topology_fingerprint_crystalnets(structure: Structure) -> str | None:
    """Hook for a true periodic quotient-graph RCSR symbol via
    CrystalNets.jl (not vendored here — requires a Julia + PyCall bridge).
    Returns None when unavailable so callers can fall back cleanly.
    """
    try:
        from juliacall import Main as jl  # optional dependency
    except ImportError:
        return None
    # Left as an integration point: write `structure` to CIF, call
    # CrystalNets.jl's `topological_genome`, parse the RCSR symbol back.
    raise NotImplementedError("wire up CrystalNets.jl here when the Julia bridge is available")


@dataclass
class BondingMetrics:
    cn_accuracy: float | None
    csm_mean_abs_diff: float | None
    distortion_pred: DistortionMetrics
    topology_match: bool
    n_aligned: int


def bonding_metrics(
    pred: Structure,
    gt: Structure,
    normalise_volume: bool,
    cnn: CrystalNN | None = None,
    compute_csm: bool = False,
) -> BondingMetrics:
    cnn = cnn or _CNN
    if normalise_volume:
        from .structure_utils import rescale_to_volume

        pred = rescale_to_volume(pred, gt.volume)

    pairs = align_sites(pred, gt)
    cn_acc = None
    csm_diff = None
    if pairs:
        cn_p = {i: cnn.get_cn(pred, i) for i, _ in pairs}
        cn_g = {j: cnn.get_cn(gt, j) for _, j in pairs}
        cn_acc = float(np.mean([cn_p[i] == cn_g[j] for i, j in pairs]))

        if compute_csm:
            diffs = []
            for i, j in pairs:
                cp, cg = chemenv_csm(pred, i), chemenv_csm(gt, j)
                if cp is not None and cg is not None:
                    diffs.append(abs(cp - cg))
            csm_diff = float(np.mean(diffs)) if diffs else None

    topo_ok = topology_fingerprint(pred, cnn) == topology_fingerprint(gt, cnn)

    return BondingMetrics(
        cn_accuracy=cn_acc,
        csm_mean_abs_diff=csm_diff,
        distortion_pred=distortion_metrics(pred, cnn),
        topology_match=topo_ok,
        n_aligned=len(pairs),
    )
