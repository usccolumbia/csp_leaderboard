"""Shared geometric primitives: reduction, standard settings, deformation gradient.

Every axis compares a *pair* of structures (prediction, ground truth), so
getting the cell convention right once, here, matters more than in a
single-structure tool: a primitive-vs-conventional or Niggli-inequivalent
pair otherwise gives meaningless strain and Wyckoff numbers.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.linalg import logm, polar


def primitive(structure: Structure, tol: float = 1e-3) -> Structure:
    return structure.get_primitive_structure(tolerance=tol)


def niggli_reduce(structure: Structure) -> Structure:
    return structure.get_reduced_structure(reduction_algo="niggli")


def canonical_cell(structure: Structure, tol: float = 1e-3) -> Structure:
    """Primitive, then Niggli-reduced. Call this before any cross-structure
    comparison (lattice error, Wyckoff matching, site alignment)."""
    return niggli_reduce(primitive(structure, tol=tol))


def to_standard_setting(structure: Structure, symprec: float = 0.01) -> Structure:
    """spglib's standardized conventional cell, at a given symprec.

    Two structures in different (but equivalent) settings — e.g. different
    origin choices for a centrosymmetric group — will otherwise fail a naive
    Wyckoff-letter comparison. Standardizing both sides first removes that
    false negative.
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    return sga.get_refined_structure()


@dataclass
class DeformationDecomposition:
    z_ratio: float
    z_mismatch: bool
    eps_vol: float | None = None       # ln det F  — signed, symmetric in over/under-prediction
    eps_shape: float | None = None     # || dev logm(U) ||_F — deviatoric Hencky strain
    aniso: np.ndarray | None = None    # eigenvalues of the Hencky strain (principal strains)
    F: np.ndarray | None = None
    R: np.ndarray | None = None
    U: np.ndarray | None = None
    basis_alignment: np.ndarray | None = None   # unimodular M applied to the predicted basis (columns), det = +1


@lru_cache(maxsize=1)
def _unimodular_candidates() -> np.ndarray:
    """All integer 3x3 matrices with entries in {-1, 0, 1} and determinant +1
    (3,480 of them): the permutations, sign changes and small shears that relate
    two equivalent Niggli-reduced bases of the same lattice. det = -1 is not
    needed because -L spans the same lattice as L."""
    grid = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=9))).reshape(-1, 3, 3)
    return grid[np.rint(np.linalg.det(grid)) == 1]


def best_basis_alignment(Lp: np.ndarray, Lg: np.ndarray) -> np.ndarray:
    """The unimodular M minimising the Hencky-strain norm of F = (Lp M) Lg^-1.

    A deformation gradient only means something when both cells use
    *corresponding* basis vectors. Two cells of the same lattice reduce to
    different, equally valid bases (axis permutations, shears) whenever the
    reduced cell has ties, and F between such bases reads as a large strain
    although nothing is deformed. Searching the finite set of equivalent
    bases and keeping the least-deforming one makes the result depend on the
    lattices only. The cost uses the eigenvalues of C = F^T F, so it is
    rotation-invariant and ties between equivalent M give the same strain.
    Columns of ``Lp`` and ``Lg`` are lattice vectors.
    """
    M = _unimodular_candidates()
    gram_p = Lp.T @ Lp
    inv_g = np.linalg.inv(Lg)
    C = inv_g.T @ (M.transpose(0, 2, 1) @ gram_p @ M) @ inv_g
    hencky = 0.5 * np.log(np.linalg.eigvalsh(C))
    return M[int(np.argmin((hencky**2).sum(axis=1)))]


def deformation_from_lattices(Lp: np.ndarray, Lg: np.ndarray) -> DeformationDecomposition:
    """Strain of ``Lp`` relative to ``Lg`` after the best basis alignment (columns = a, b, c)."""
    M = best_basis_alignment(Lp, Lg)
    F = (Lp @ M) @ np.linalg.inv(Lg)

    R, U = polar(F, side="right")  # F = R U, U symmetric positive-definite
    eps_vol = float(np.log(np.linalg.det(F)))

    H = logm(U).real  # Hencky (true) strain tensor
    dev_H = H - np.trace(H) / 3.0 * np.eye(3)
    return DeformationDecomposition(
        z_ratio=1.0,
        z_mismatch=False,
        eps_vol=eps_vol,
        eps_shape=float(np.linalg.norm(dev_H, ord="fro")),
        aniso=np.linalg.eigvalsh(H),
        F=F,
        R=R,
        U=U,
        basis_alignment=M,
    )


def deformation_gradient(pred: Structure, gt: Structure, z_tol: float = 0.02) -> DeformationDecomposition:
    """F = L_pred M L_gt^-1 in column-vector convention, then F = R U (polar decomposition).

    Uses matrix columns = lattice vectors (a, b, c), i.e. the transpose of
    pymatgen's row-vector ``Lattice.matrix``, so that F acts on fractional
    coordinates the way a deformation gradient conventionally does. ``M`` is
    the best unimodular basis alignment (see ``best_basis_alignment``).
    """
    P = canonical_cell(pred)
    G = canonical_cell(gt)

    z_ratio = len(P) / len(G)
    if abs(z_ratio - 1.0) > z_tol:
        # supercell/subcell mismatch: scored as its own bucket, never as strain
        return DeformationDecomposition(z_ratio=z_ratio, z_mismatch=True)

    d = deformation_from_lattices(P.lattice.matrix.T, G.lattice.matrix.T)  # columns = a, b, c
    d.z_ratio = z_ratio
    return d


def rescale_to_volume(structure: Structure, target_volume: float) -> Structure:
    """Isotropic volume rescale, fractional coordinates fixed."""
    scale = (target_volume / structure.volume) ** (1.0 / 3.0)
    out = structure.copy()
    out.scale_lattice(structure.volume * scale**3)
    return out
