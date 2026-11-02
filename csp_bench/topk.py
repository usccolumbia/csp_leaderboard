"""Top-1 / top-K aggregation.

Every metric elsewhere in this package is defined on ONE (prediction,
ground truth) pair. With k samples per formula each needs a reduction, and
per the protocol doc the choice of reduction is itself a result, not a
bookkeeping detail:

    M@1   energy-ranked best sample   — the deployed pipeline
    M@k   oracle over the dedup'd set — generator capability
    rank  position of the best sample in the ranked list — ranker quality

The gap M@k - M@1 is "ranker headroom": the accuracy a perfect energy
model would unlock with no change to the generator. It separates two
failure modes the field conflates — never generated, vs. generated and
mis-ranked — but see ``ranker_sensitivity`` below: as measured with one
frozen MLIP, that gap is a claim about *that MLIP's* ranking, not about
energy ranking in general.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pymatgen.core import Structure

from .bonding import bonding_metrics
from .cascade import assign_bucket, bucket_rank
from .energy import energy_metrics
from .matcher import FROZEN_MATCHER, match, match_distance
from .mlip import MLIP
from .symmetry import symmetry_ladder

RESOLUTION_LIMIT_EV = 0.010  # eV/atom; below this, no realistic energy model can resolve the ranking


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


@dataclass
class CandidatePool:
    unique: list[Structure]
    n_raw: int
    effective_k_ratio: float  # len(unique) / n_raw — a collapse ratio near 0 means the model has far less than k real ideas


def prepare_candidates(samples: list[Structure], matcher=None) -> CandidatePool:
    """Dedupe with the SAME matcher and tolerances used for scoring — a
    different, looser dedupe tolerance would silently inflate effective k.
    """
    matcher = matcher or FROZEN_MATCHER
    unique: list[Structure] = []
    for s in samples:
        if not any(matcher.fit(s, u) for u in unique):
            unique.append(s)
    ratio = len(unique) / len(samples) if samples else 0.0
    return CandidatePool(unique=unique, n_raw=len(samples), effective_k_ratio=ratio)


# ---------------------------------------------------------------------------
# Continuous metrics on ONE named candidate (never min-over-k independently
# per axis — that describes a structure no sample actually is)
# ---------------------------------------------------------------------------


@dataclass
class AllAxes:
    symmetry: object
    lattice: object
    bonding: object
    energy: object


def all_axes(pred: Structure, gt: Structure, mlip: MLIP, symprec: float = 0.1):
    from .lattice import lattice_error

    return AllAxes(
        symmetry=symmetry_ladder(pred, gt, symprec=symprec),
        lattice=lattice_error(pred, gt),
        bonding=bonding_metrics(pred, gt, normalise_volume=True),
        energy=energy_metrics(pred, gt, mlip),
    )


# ---------------------------------------------------------------------------
# The per-formula reduction
# ---------------------------------------------------------------------------


@dataclass
class ReducedResult:
    match_at_1: bool
    match_at_k: bool
    first_hit: int | None          # rank (0-indexed) of the first matching candidate, or None
    cont_at_1: AllAxes
    cont_best: AllAxes
    bucket_at_1: str
    bucket_at_k: str
    pool: CandidatePool
    ranked_energies: list[float]   # per-atom energies in ranked order, for the resolution-limit check


def reduce_over_k(
    samples: list[Structure],
    gt: Structure,
    mlip: MLIP,
    matcher=None,
    dedupe: bool = True,
) -> ReducedResult:
    matcher = matcher or FROZEN_MATCHER
    pool = prepare_candidates(samples, matcher) if dedupe else CandidatePool(samples, len(samples), 1.0)
    candidates = pool.unique

    # Rank by the frozen MLIP's relaxed energy per atom — "the deployed pipeline"
    relaxed = [mlip.relax(c).energy_per_atom for c in candidates]
    order = sorted(range(len(candidates)), key=lambda i: relaxed[i])
    ranked = [candidates[i] for i in order]
    ranked_energies = [relaxed[i] for i in order]

    top1 = ranked[0]
    matches = [match(c, gt, matcher) for c in candidates]
    match_at_1 = match(top1, gt, matcher)
    match_at_k = any(matches)
    first_hit = next((rank for rank, i in enumerate(order) if matches[i]), None)

    distances = [match_distance(c, gt, matcher) for c in candidates]
    best_match = candidates[distances.index(min(distances))]

    cont_at_1 = all_axes(top1, gt, mlip)
    cont_best = all_axes(best_match, gt, mlip)

    bucket_at_1 = assign_bucket(top1, gt, mlip, matcher).bucket
    bucket_traces = [assign_bucket(c, gt, mlip, matcher) for c in candidates]
    bucket_at_k = min((t.bucket for t in bucket_traces), key=bucket_rank)

    return ReducedResult(
        match_at_1=match_at_1,
        match_at_k=match_at_k,
        first_hit=first_hit,
        cont_at_1=cont_at_1,
        cont_best=cont_best,
        bucket_at_1=bucket_at_1,
        bucket_at_k=bucket_at_k,
        pool=pool,
        ranked_energies=ranked_energies,
    )


# ---------------------------------------------------------------------------
# pass@k without resampling
# ---------------------------------------------------------------------------


def pass_at_k_estimate(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): generate n > k samples
    once, estimate match@k for every k <= n from that single pool instead
    of regenerating. ``c`` = number of samples among n that match.

        match@k = 1 - C(n-c, k) / C(n, k)

    The shape of the resulting curve vs. k is itself diagnostic: saturating
    by k=3 means the model has one idea per formula; still climbing at
    k=20 means coverage is there and ranking is the binding constraint.
    """
    if k > n:
        raise ValueError(f"k={k} exceeds pool size n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_curve(n: int, c: int, ks: list[int] | None = None) -> dict[int, float]:
    ks = ks or list(range(1, n + 1))
    return {k: pass_at_k_estimate(n, c, k) for k in ks}


# ---------------------------------------------------------------------------
# Ranker sensitivity — does "ranker headroom" depend on WHICH MLIP ranks?
# ---------------------------------------------------------------------------


@dataclass
class RankerSensitivity:
    top1_primary: Structure
    top1_secondary: Structure
    agree: bool                # same structure selected as top-1 by both rankers
    match_primary: bool
    match_secondary: bool


def ranker_sensitivity(
    samples: list[Structure],
    gt: Structure,
    primary: MLIP,
    secondary: MLIP,
    matcher=None,
) -> RankerSensitivity:
    """Rerank the SAME candidate pool with a second, independent MLIP
    (e.g. UMA) while every diagnostic energy elsewhere stays on the
    primary (MatterSim). Close agreement between ``match_primary`` and
    ``match_secondary`` across a test set means ranker headroom reflects a
    real generator/ranking limit; a systematic gap means the reported
    M@1 is partly an artifact of the primary MLIP's ranking idiosyncrasies
    and should be reported as such (see the protocol's "Ranker
    sensitivity" section).
    """
    matcher = matcher or FROZEN_MATCHER
    pool = prepare_candidates(samples, matcher).unique

    e_primary = [primary.relax(c).energy_per_atom for c in pool]
    e_secondary = [secondary.relax(c).energy_per_atom for c in pool]

    top1_p = pool[e_primary.index(min(e_primary))]
    top1_s = pool[e_secondary.index(min(e_secondary))]

    return RankerSensitivity(
        top1_primary=top1_p,
        top1_secondary=top1_s,
        agree=match(top1_p, top1_s, matcher),
        match_primary=match(top1_p, gt, matcher),
        match_secondary=match(top1_s, gt, matcher),
    )


# ---------------------------------------------------------------------------
# Ranker resolution limit
# ---------------------------------------------------------------------------


@dataclass
class ResolutionLimit:
    applicable: bool              # True only when correct structure is present but mis-ranked
    delta_e: float | None         # eV/atom gap between top-ranked wrong candidate and the correct one
    resolvable: bool | None       # False = below the MLIP noise floor; no ranker fix can help


def resolution_limit(result: ReducedResult, candidates: list[Structure], gt: Structure, mlip: MLIP, matcher=None) -> ResolutionLimit:
    matcher = matcher or FROZEN_MATCHER
    if result.match_at_1 or not result.match_at_k:
        return ResolutionLimit(applicable=False, delta_e=None, resolvable=None)

    # top-ranked wrong candidate = rank-0 by energy among non-matching ones
    wrong = [c for c in candidates if not match(c, gt, matcher)]
    wrong_energies = [(mlip.relax(c).energy_per_atom, c) for c in wrong]
    e_wrong_top, _ = min(wrong_energies, key=lambda t: t[0])

    correct = next(c for c in candidates if match(c, gt, matcher))
    e_correct = mlip.relax(correct).energy_per_atom

    delta = e_wrong_top - e_correct
    return ResolutionLimit(applicable=True, delta_e=delta, resolvable=abs(delta) >= RESOLUTION_LIMIT_EV)
