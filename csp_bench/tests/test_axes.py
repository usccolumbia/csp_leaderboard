"""Fast tests for Axes A-C and top-K dedupe/pass@k — no MLIP required."""
import math

import numpy as np
from pymatgen.core import Lattice, Structure

from csp_bench import bonding, matcher, symmetry, topk
from csp_bench.lattice import cubic_deviation, lattice_error


def _toy_binary(a=5.64):
    """CsCl-type: simple cubic Bravais lattice, 2 sites, space group 221.
    Deliberately NOT rocksalt (that needs the FCC-symmetric arrangement
    below) — used where the exact space group doesn't matter."""
    return Structure(Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _rocksalt(a=5.64):
    """True rocksalt, Fm-3m (225), built via the spacegroup constructor so
    it actually carries FCC symmetry (8 sites in the conventional cell)."""
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_symmetry_ladder_identical():
    gt = _rocksalt()
    lad = symmetry.symmetry_ladder(gt, gt.copy(), symprec=0.01)
    assert lad.system and lad.sg and lad.wyckoff and lad.decorated
    assert lad.pred_sg_number == lad.gt_sg_number == 225


def test_symmetry_slack_detects_near_miss():
    gt = _toy_binary()
    pred = gt.copy()
    pred.translate_sites([1], [0.03, 0.0, 0.0], frac_coords=True)
    # tight symprec should no longer see the parent group; loose should
    assert not symmetry.symmetry_ladder(pred, gt, symprec=0.01).sg
    assert symmetry.symmetry_ladder(pred, gt, symprec=0.5).sg
    assert symmetry.symmetry_slack(pred, gt) == 1


def test_crystal_system_ranges():
    assert symmetry.crystal_system(1) == "triclinic"
    assert symmetry.crystal_system(225) == "cubic"
    assert symmetry.crystal_system(75) == "tetragonal"


def test_lattice_error_identical_is_zero():
    gt = _toy_binary()
    err = lattice_error(gt, gt.copy())
    assert not err.z_mismatch
    assert abs(err.eps_vol) < 1e-8
    assert abs(err.eps_shape) < 1e-8


def test_lattice_error_volume_scale_is_signed_symmetric():
    gt = _toy_binary(a=5.0)
    shrunk = _toy_binary(a=5.0 * 0.5 ** (1 / 3))   # half the volume
    grown = _toy_binary(a=5.0 * 2 ** (1 / 3))      # double the volume
    e_shrunk = lattice_error(shrunk, gt).eps_vol
    e_grown = lattice_error(grown, gt).eps_vol
    # log space: an under- and an over-prediction of the same factor are
    # exactly opposite in sign — this is the whole point of eps_vol = ln det F
    # (a raw V_pred/V_gt ratio would read 0.5 and 2.0: not symmetric, and
    # would silently bias a mean taken over a test set).
    assert math.isclose(e_shrunk, -e_grown, rel_tol=1e-9)


def test_lattice_error_flags_genuine_z_mismatch():
    gt = _toy_binary()
    # A 4-site structure whose translational symmetry is deliberately broken
    # (the second formula unit's Cl sits at 0.8, not 0.75) so it does NOT
    # reduce back to a 2-atom primitive cell — a genuinely different
    # periodicity from gt, not merely a redundant supercell of it.
    pred = Structure(
        Lattice.orthorhombic(2 * 5.64, 5.64, 5.64),
        ["Na", "Cl", "Na", "Cl"],
        [[0, 0, 0], [0.25, 0.5, 0.5], [0.5, 0, 0], [0.8, 0.5, 0.5]],
    )
    err = lattice_error(pred, gt)
    assert err.z_mismatch
    assert err.z_ratio == 2.0
    assert err.eps_vol is None  # never scored as strain — see doc's "Z-mismatch is its own bucket"
    assert err.cdm_pred is None and err.cdm_gt is None and err.delta_cdm is None


def test_lattice_error_supercell_of_same_structure_is_not_a_mismatch():
    # An EXACT supercell reduces back to the same primitive cell as gt, so
    # this is correctly NOT a z-mismatch: the prediction is the same
    # structure, just described with a redundant repeat unit.
    gt = _toy_binary()
    supercell = gt.copy()
    supercell.make_supercell([2, 1, 1])
    err = lattice_error(supercell, gt)
    assert not err.z_mismatch


def test_cubic_deviation_cubic_is_zero_and_scale_invariant():
    assert abs(cubic_deviation(Lattice.cubic(1.0))) < 1e-12
    assert abs(cubic_deviation(Lattice.cubic(7.3))) < 1e-12
    # unitless: tripling every edge leaves the shape, hence the value, unchanged
    assert math.isclose(
        cubic_deviation(Lattice.tetragonal(1.0, 2.0)),
        cubic_deviation(Lattice.tetragonal(3.0, 6.0)),
        rel_tol=1e-12,
    )


def test_cubic_deviation_matches_hand_derived_values():
    # Hand-evaluated from Bernier et al. (arXiv:2508.01177) Eq. 5, not by
    # running the function under test.
    # Tetragonal a=b=1, c=2: the ab face is a square (0); each of the two
    # square-less faces gives |1/sqrt5 - 1/sqrt2| + |2/sqrt5 - 1/sqrt2| = 1/sqrt5,
    # so CDM = (2/3)/sqrt5.
    assert math.isclose(cubic_deviation(Lattice.tetragonal(1.0, 2.0)), 2 / (3 * math.sqrt(5)), rel_tol=1e-12)
    # Hexagonal a=b=c, gamma=120: only the ab face deviates, giving
    # (1/3) * 2 * |1/sqrt3 - 1/sqrt2| = (sqrt2 - 2/sqrt3) / 3 ~= 0.0865. (The
    # paper's Eq. 6 prints "~= 0.0809" next to this same closed form; 0.0809
    # is actually the value of its rhombohedral setting, alpha ~= 97.2 deg.)
    hexagonal = Lattice.from_parameters(1, 1, 1, 90, 90, 120)
    assert math.isclose(cubic_deviation(hexagonal), (math.sqrt(2) - 2 / math.sqrt(3)) / 3, rel_tol=1e-12)
    # fcc PRIMITIVE cell (60 deg): nonzero, unlike the conventional cubic cell.
    fcc_primitive = Lattice.from_parameters(1, 1, 1, 60, 60, 60)
    assert math.isclose(cubic_deviation(fcc_primitive), math.sqrt(2) - 2 / math.sqrt(3), rel_tol=1e-12)


def test_cubic_deviation_permutation_invariant_and_bounded():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(500):
        a, b, c = rng.uniform(0.3, 3.0, 3)
        al, be, ga = rng.uniform(40.0, 140.0, 3)
        lat = Lattice.from_parameters(a, b, c, al, be, ga)
        if not (np.isfinite(lat.volume) and lat.volume > 1e-6):
            continue  # this angle triple cannot form a parallelepiped
        v = cubic_deviation(lat)
        # relabel the axes a'=b, b'=c, c'=a; each angle follows its pair of axes
        v_perm = cubic_deviation(Lattice.from_parameters(b, c, a, be, ga, al))
        assert math.isclose(v, v_perm, rel_tol=1e-9, abs_tol=1e-12)
        assert -1e-12 <= v <= 1.0 + 1e-12
        checked += 1
    assert checked > 100


def test_lattice_error_cdm_identical_is_zero_delta():
    gt = _toy_binary()
    err = lattice_error(gt, gt.copy())
    assert abs(err.delta_cdm) < 1e-12
    assert math.isclose(err.cdm_pred, err.cdm_gt, abs_tol=1e-12)


def test_lattice_error_cdm_ignores_pure_dilation():
    # The shape metric must be orthogonal to the volume error: a uniformly
    # scaled but otherwise identical lattice is wrong in eps_vol only.
    gt = _toy_binary(a=5.0)
    grown = _toy_binary(a=5.0 * 2 ** (1 / 3))
    err = lattice_error(grown, gt)
    assert abs(err.eps_vol) > 0.5
    assert abs(err.delta_cdm) < 1e-12


def test_lattice_error_delta_cdm_is_signed_and_antisymmetric():
    cubic = _toy_binary(a=5.0)
    stretched = Structure(Lattice.tetragonal(5.0, 6.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    too_distorted = lattice_error(stretched, cubic).delta_cdm   # pred less cubic than gt
    too_cubic = lattice_error(cubic, stretched).delta_cdm       # pred more cubic than gt
    assert too_distorted > 0 > too_cubic
    assert math.isclose(too_distorted, -too_cubic, rel_tol=1e-12)


def test_lattice_error_cdm_uses_the_canonical_primitive_cell():
    # Documented convention: CDM is evaluated on the primitive, Niggli-reduced
    # cell shared with eps_shape, so an fcc crystal reads ~0.2595 rather than
    # the 0 of its conventional cubic cell. Pinned so a change is deliberate.
    rs = _rocksalt()
    err = lattice_error(rs, rs.copy())
    assert math.isclose(err.cdm_gt, math.sqrt(2) - 2 / math.sqrt(3), rel_tol=1e-6)
    assert abs(cubic_deviation(rs.lattice)) < 1e-12  # the conventional cell itself is cubic


def _naive_eps_shape(Lp, Lg):
    """The pre-alignment formula: F straight from the two bases as given."""
    from scipy.linalg import logm, polar
    _, U = polar(Lp @ np.linalg.inv(Lg), side="right")
    H = logm(U).real
    return float(np.linalg.norm(H - np.trace(H) / 3.0 * np.eye(3), ord="fro"))


_EQUIVALENT_BASES = [
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),   # axis permutation
    np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float),   # shear a, a+b, c
    np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=float),   # shear c into a and b
]


def test_deformation_ignores_equivalent_basis_choice():
    from csp_bench.structure_utils import deformation_from_lattices

    Lg = Lattice.from_parameters(4.0, 5.0, 6.5, 80.0, 95.0, 100.0).matrix.T  # columns = a, b, c
    for M in _EQUIVALENT_BASES:
        Lp = Lg @ M  # the SAME lattice written in another basis
        assert _naive_eps_shape(Lp, Lg) > 0.05  # the bug: reads as strain although nothing is deformed
        d = deformation_from_lattices(Lp, Lg)
        assert d.eps_shape < 1e-9
        assert abs(d.eps_vol) < 1e-9


def test_deformation_recovers_known_strain_in_any_basis():
    from scipy.spatial.transform import Rotation

    from csp_bench.structure_utils import deformation_from_lattices

    Lg = Lattice.from_parameters(4.0, 5.0, 6.5, 80.0, 95.0, 100.0).matrix.T
    Q = Rotation.from_euler("xyz", [31.0, -17.0, 64.0], degrees=True).as_matrix()
    stretch = np.diag([1.2, 1.0, 1.0])
    # H = diag(ln 1.2, 0, 0); its deviatoric part has Frobenius norm ln(1.2) * sqrt(6) / 3
    expected = math.log(1.2) * math.sqrt(6.0) / 3.0
    for M in [np.eye(3)] + _EQUIVALENT_BASES:
        d = deformation_from_lattices(Q @ stretch @ Lg @ M, Lg)
        assert math.isclose(d.eps_shape, expected, rel_tol=1e-6)
        assert math.isclose(d.eps_vol, math.log(1.2), rel_tol=1e-6)


def test_lattice_error_equivalent_cell_choice_is_not_strain():
    gt = Structure(Lattice.hexagonal(3.0, 5.0), ["Zn", "S"], [[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.5]])
    pred = gt.copy()
    pred.make_supercell([[1, 0, 0], [1, 1, 0], [0, 0, 1]])  # cell (a, a+b, c): same crystal, gamma = 60 deg
    err = lattice_error(pred, gt)
    assert not err.z_mismatch
    assert err.eps_shape < 1e-6 and abs(err.eps_vol) < 1e-9 and abs(err.delta_cdm) < 1e-9


def test_matcher_frozen_tolerances_and_distance():
    gt = _toy_binary()
    pred = gt.copy()
    pred.translate_sites([1], [0.02, 0.0, 0.0], frac_coords=True)
    assert matcher.match(pred, gt)
    assert matcher.match_distance(pred, gt) > 0

    wrong_composition = _toy_binary()
    wrong_composition.replace(1, "Na")  # Na-Na is a different composition, guaranteed no match
    assert not matcher.match(wrong_composition, gt)
    assert matcher.match_distance(wrong_composition, gt) == float("inf")


def test_bonding_metrics_identical_structure():
    gt = _toy_binary()
    bm = bonding.bonding_metrics(gt, gt.copy(), normalise_volume=False)
    assert bm.cn_accuracy == 1.0
    assert bm.topology_match
    assert bm.n_aligned == 2


def test_prepare_candidates_dedupes_with_frozen_matcher():
    gt = _toy_binary()
    near = gt.copy()
    near.translate_sites([1], [0.01, 0.0, 0.0], frac_coords=True)
    wrong_composition = _toy_binary()
    wrong_composition.replace(1, "Na")

    pool = topk.prepare_candidates([gt, near, near.copy(), wrong_composition])
    assert len(pool.unique) == 2  # {gt≈near} collapse, wrong_composition stays separate
    assert pool.n_raw == 4
    assert pool.effective_k_ratio == 0.5


def test_pass_at_k_estimate_matches_hand_computation():
    # n=5 samples, c=2 correct: pass@1 should equal c/n
    assert math.isclose(topk.pass_at_k_estimate(5, 2, 1), 2 / 5, rel_tol=1e-9)
    # pass@n is always 1 when at least one sample is correct
    assert topk.pass_at_k_estimate(5, 2, 5) == 1.0
    # monotonic non-decreasing in k
    curve = topk.pass_at_k_curve(10, 3)
    values = [curve[k] for k in sorted(curve)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_pass_at_k_zero_correct_is_zero():
    assert topk.pass_at_k_estimate(10, 0, 5) == 0.0
