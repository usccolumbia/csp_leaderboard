"""End-to-end top-1/top-K reduction against a real MLIP backend.

Slow (loads a network potential, runs relaxations) — run explicitly:

    pytest csp_bench/tests/test_topk_with_mlip.py -m slow

Uses CHGNet rather than MatterSim (the study's frozen choice) purely
because CHGNet's weights are already cached in this environment; swap
``get_mlip("chgnet")`` for ``get_mlip("mattersim")`` once that download is
available. The logic under test doesn't depend on which backend it is —
that's the point of the ``MLIP`` abstraction in ``csp_bench.mlip``.
"""
import pytest
from pymatgen.core import Lattice, Structure

from csp_bench import topk
from csp_bench.mlip import get_mlip

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def mlip():
    return get_mlip("chgnet")


def _toy_binary(a=5.64):
    return Structure(Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_reduce_over_k_matching_candidate_present(mlip):
    gt = _toy_binary()
    near = gt.copy()
    near.translate_sites([1], [0.02, 0.0, 0.0], frac_coords=True)
    wrong_composition = _toy_binary()
    wrong_composition.replace(1, "Na")

    result = topk.reduce_over_k([gt, near, wrong_composition], gt, mlip)

    assert result.match_at_k is True
    assert result.match_at_1 is True  # gt itself is in the pool and is lowest-energy
    assert result.first_hit == 0
    assert result.bucket_at_1 == "F0"
    assert result.bucket_at_k == "F0"


def test_reduce_over_k_no_match_falls_back_to_taxonomy(mlip):
    gt = _toy_binary()
    only_wrong = _toy_binary()
    only_wrong.replace(1, "Na")

    result = topk.reduce_over_k([only_wrong], gt, mlip)

    assert result.match_at_1 is False
    assert result.match_at_k is False
    assert result.first_hit is None
    assert result.bucket_at_1 != "F0"


def test_ranker_sensitivity_self_agreement(mlip):
    gt = _toy_binary()
    near = gt.copy()
    near.translate_sites([1], [0.02, 0.0, 0.0], frac_coords=True)

    # Same backend as both "rankers": must trivially agree with itself —
    # a regression here means ranker_sensitivity's two energy passes are
    # somehow not deterministic/comparable.
    rs = topk.ranker_sensitivity([gt, near], gt, mlip, mlip)
    assert rs.agree
    assert rs.match_primary == rs.match_secondary


def test_resolution_limit_not_applicable_when_top1_already_matches(mlip):
    gt = _toy_binary()
    result = topk.reduce_over_k([gt], gt, mlip)
    rl = topk.resolution_limit(result, [gt], gt, mlip)
    assert rl.applicable is False
