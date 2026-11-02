"""Axis B — lattice and unit-cell error.

Not one more axis alongside bonding: lattice error contaminates bonding
(a too-small cell fakes short contacts and imaginary phonons; a too-large
cell fakes broken coordination and wrong topology), so it is measured first
and normalised out before Axis C is trusted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pymatgen.core import Lattice, Structure

from .structure_utils import DeformationDecomposition, canonical_cell, deformation_gradient

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _face_deviation(p: float, q: float, angle_deg: float) -> float:
    # m is the longer diagonal of the p-q face (Bernier et al. Eq. 3, which
    # collapses to |cos| because cos(gamma - 180 deg) = -cos(gamma)). In a
    # square p == q and the angle is 90 deg, so p/m == q/m == 1/sqrt(2).
    m = math.sqrt(p * p + q * q + 2.0 * p * q * abs(math.cos(math.radians(angle_deg))))
    return abs(p / m - _INV_SQRT2) + abs(q / m - _INV_SQRT2)


def cubic_deviation(lattice: Lattice) -> float:
    """Cubic deviation metric (CDM) of a single unit cell, in [0, 1].

    Bernier et al., arXiv:2508.01177, Eq. 5: the mean over the three unique
    faces of how far each face's side/diagonal ratios are from the 1/sqrt(2)
    of a square. Uses only the six lattice parameters, is unitless and
    scale-invariant, is unchanged by permuting a, b, c, and is exactly 0 if
    and only if the cell is cubic.

    This describes ONE cell, so it depends on which cell you pass: a
    conventional fcc cell gives 0 but its primitive cell (alpha = 60 deg)
    gives sqrt(2) - 2/sqrt(3) ~= 0.2595. It is also not injective: distinct
    shapes can share a value, so a CDM difference of 0 does not mean the
    cells agree. Use it next to ``eps_shape``, never instead of it.
    """
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles
    return (
        _face_deviation(a, b, gamma)
        + _face_deviation(a, c, beta)
        + _face_deviation(b, c, alpha)
    ) / 3.0


@dataclass
class LatticeError:
    z_mismatch: bool
    z_ratio: float
    eps_vol: float | None      # signed log-volume error; mean over a set = density bias
    eps_shape: float | None    # deviatoric Hencky strain norm; scale-independent cell-shape error
    aniso: object | None       # principal strains (eigenvalues of Hencky strain)
    cdm_pred: float | None = None    # cubic deviation of the predicted canonical cell
    cdm_gt: float | None = None      # cubic deviation of the reference canonical cell
    delta_cdm: float | None = None   # cdm_pred - cdm_gt; signed, > 0 = prediction less cubic than the reference


def lattice_error(pred: Structure, gt: Structure, z_tol: float = 0.02) -> LatticeError:
    """Signed ``delta_cdm`` is taken on the same canonical (primitive,
    Niggli-reduced) cells as ``eps_shape``, so both describe the same pair
    of cells. It is None under a Z mismatch, like the other strain terms.
    Unlike the unsigned ``eps_shape``, its sign says which way the model
    errs, and a mean over a set is a bias (e.g. systematically too cubic).
    """
    d: DeformationDecomposition = deformation_gradient(pred, gt, z_tol=z_tol)
    cdm_pred = cdm_gt = delta_cdm = None
    if not d.z_mismatch:
        cdm_pred = cubic_deviation(canonical_cell(pred).lattice)
        cdm_gt = cubic_deviation(canonical_cell(gt).lattice)
        delta_cdm = cdm_pred - cdm_gt
    return LatticeError(
        z_mismatch=d.z_mismatch,
        z_ratio=d.z_ratio,
        eps_vol=d.eps_vol,
        eps_shape=d.eps_shape,
        aniso=d.aniso,
        cdm_pred=cdm_pred,
        cdm_gt=cdm_gt,
        delta_cdm=delta_cdm,
    )


def volume_per_atom_prior(structure: Structure, ionic_radii: dict[str, float] | None = None) -> float:
    """Composition-only volume prior: sum of cubed ionic/atomic radii per atom.

    Used to test whether a model's predicted volume tracks a density prior
    (``eps_vol`` bias near zero) or is a flat, uninformed scatter (see the
    "design question" discussion in the protocol: does the lattice
    parameterisation matter?). ``ionic_radii`` defaults to Shannon-ish
    single-valence atomic radii pulled from pymatgen's Element data.
    """
    from pymatgen.core.periodic_table import Element

    total = 0.0
    for site in structure:
        el = Element(site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie))
        r = None
        if ionic_radii and str(site.specie) in ionic_radii:
            r = ionic_radii[str(site.specie)]
        elif el.atomic_radius:
            r = float(el.atomic_radius)
        else:
            r = 1.5  # fallback, Å
        total += r**3
    return total / len(structure)
