"""The single frozen StructureMatcher used by every axis and every model.

Tolerances (ltol 0.3, stol 0.5, angle_tol 10) match the CSP-literature
default so numbers stay comparable across papers. Changing these anywhere
except here breaks that comparability — import ``FROZEN_MATCHER`` rather
than constructing a new one.
"""
from __future__ import annotations

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

FROZEN_MATCHER = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True)


def match(pred: Structure, gt: Structure, matcher: StructureMatcher | None = None) -> bool:
    matcher = matcher or FROZEN_MATCHER
    # StructureMatcher.fit returns numpy.bool_; coerce so every caller can
    # rely on this actually being `bool` (e.g. `is True`, JSON serialization).
    return bool(matcher.fit(pred, gt))


def match_distance(pred: Structure, gt: Structure, matcher: StructureMatcher | None = None) -> float:
    """A continuous "how close" distance for ranking candidates by match
    quality, not just pass/fail: the matcher's RMS displacement when a fit
    is found, else +inf. Used by ``topk.reduce_over_k`` to pick the
    "match-best" candidate among k when more than one matches (or none do).
    """
    matcher = matcher or FROZEN_MATCHER
    result = matcher.get_rms_dist(pred, gt)
    if result is None:
        return float("inf")
    rms, _max_dist = result
    return float(rms)
