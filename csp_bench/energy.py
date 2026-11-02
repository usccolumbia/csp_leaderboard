"""Axis D — energy and dynamical stability.

Ground truth exists here (unlike unconditional-generation benchmarks), so
energy and phonons are *instruments* for diagnosing a prediction against a
known answer, not free-standing scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Structure

from .bonding import align_sites
from .mlip import MLIP

FOUND_LOWER_THRESHOLD_EV = -0.005  # eV/atom; below this, call it a better polymorph, not an error


@dataclass
class EnergyMetrics:
    dE: float                 # E(relaxed pred) - E(relaxed gt), per atom — primary metric
    d_hull: float | None      # paired e_above_hull delta, when a phase-diagram context is supplied
    found_lower: bool         # pred relaxed BELOW the reference entry — not a failure, its own bucket (F8)


def energy_metrics(pred: Structure, gt: Structure, mlip: MLIP, hull_context=None) -> EnergyMetrics:
    """``hull_context``, if given, is anything exposing
    ``e_above_hull(structure, mlip) -> float`` (e.g. a wrapper around a
    pymatgen ``PhaseDiagram`` built with MLIP-relaxed competing phases).
    Left optional because building a consistent MLIP-based hull is a
    separate, expensive per-composition step outside this module's scope.
    """
    Ep = mlip.relax(pred).energy_per_atom
    Eg = mlip.relax(gt).energy_per_atom
    dE = Ep - Eg

    d_hull = None
    if hull_context is not None:
        d_hull = hull_context.e_above_hull(pred, mlip) - hull_context.e_above_hull(gt, mlip)

    return EnergyMetrics(dE=dE, d_hull=d_hull, found_lower=(dE < FOUND_LOWER_THRESHOLD_EV))


# ---------------------------------------------------------------------------
# Phonons
# ---------------------------------------------------------------------------


def auto_supercell(structure: Structure, target_atoms: int = 150) -> tuple[int, int, int]:
    """Smallest diagonal supercell multiplier pushing the cell up to
    roughly ``target_atoms`` sites, capped per axis so a very anisotropic
    primitive cell doesn't blow up along one direction."""
    n = len(structure)
    reps = max(1, round((target_atoms / n) ** (1 / 3)))
    return (min(reps, 4), min(reps, 4), min(reps, 4))


@dataclass
class PhononResult:
    frequencies: np.ndarray            # (n_qpoints, n_bands), THz
    eigenvectors: np.ndarray           # (n_qpoints, n_bands, n_atoms, 3) complex
    qpoints: np.ndarray
    min_freq: float
    imaginary_modes: list              # [(freq, eigenvector[n_atoms,3]), ...] at q=Gamma

    def stable_at(self, threshold_thz: float) -> bool:
        return self.min_freq > threshold_thz


def phonon_metrics(
    structure: Structure,
    mlip: MLIP,
    mesh=(6, 6, 6),
    supercell: tuple[int, int, int] | None = None,
    fmax: float = 0.001,
) -> PhononResult:
    """Tight pre-phonon relaxation (fmax=0.001 eV/Å by default — loose
    residual forces otherwise produce spurious imaginary modes and
    confound Axis D with the post-processing cascade), full-BZ mesh (never
    Gamma-only, which misses zone-boundary instabilities), acoustic sum
    rule enforced on the force constants.
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    relaxed = mlip.relax(structure, fmax=fmax, cell_relax=True).structure
    supercell = supercell or auto_supercell(relaxed)

    unit = PhonopyAtoms(
        symbols=[str(s.specie) for s in relaxed],
        cell=relaxed.lattice.matrix,
        scaled_positions=relaxed.frac_coords,
        masses=[float(s.specie.atomic_mass) for s in relaxed],  # phonopy lacks masses for e.g. Pu
    )
    phonon = Phonopy(unit, supercell_matrix=np.diag(supercell))
    phonon.generate_displacements(distance=0.01)

    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    force_sets = []
    for sc in phonon.supercells_with_displacements:
        pmg_struct = Structure(
            lattice=sc.cell,
            species=sc.symbols,
            coords=sc.scaled_positions,
            coords_are_cartesian=False,
        )
        atoms = adaptor.get_atoms(pmg_struct)
        atoms.calc = mlip.calculator
        force_sets.append(atoms.get_forces())

    phonon.forces = force_sets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()  # enforces the acoustic sum rule

    phonon.run_mesh(mesh, with_eigenvectors=True, is_gamma_center=True)
    mesh_dict = phonon.get_mesh_dict()
    freqs = mesh_dict["frequencies"]            # (n_qpoints, n_bands)
    eigvecs = mesh_dict["eigenvectors"]          # (n_qpoints, n_bands, 3*n_atoms) complex
    qpoints = mesh_dict["qpoints"]

    n_atoms = freqs.shape[1] // 3  # phonopy may reduce a non-primitive cell
    eigvecs = eigvecs.reshape(eigvecs.shape[0], eigvecs.shape[1], n_atoms, 3)

    gamma_idx = int(np.argmin(np.linalg.norm(qpoints, axis=1)))
    imaginary = [
        (float(freqs[gamma_idx, b]), eigvecs[gamma_idx, b])
        for b in range(freqs.shape[1])
        if freqs[gamma_idx, b] < 0
    ]

    return PhononResult(
        frequencies=freqs,
        eigenvectors=eigvecs,
        qpoints=qpoints,
        min_freq=float(freqs.min()),
        imaginary_modes=imaginary,
    )


def soft_mode_overlap(pred: Structure, gt: Structure, phonons: PhononResult) -> float | None:
    """Overlap of the (ground-truth minus prediction) displacement field
    with each imaginary-mode eigenvector of ``pred``, maximised over modes.

    High overlap => the prediction landed on the high-symmetry parent and
    its own instability points directly at the true ground state (a
    displacive transition, e.g. cubic to tetragonal) — a cheap fix:
    post-process by following the soft mode and re-relaxing.
    """
    if not phonons.imaginary_modes:
        return None

    pairs = align_sites(pred, gt)
    if not pairs:
        return None

    u = np.zeros((len(pred), 3))
    for i, j in pairs:
        d_frac = gt.frac_coords[j] - pred.frac_coords[i]
        d_frac -= np.round(d_frac)  # minimum image
        u[i] = pred.lattice.get_cartesian_coords(d_frac)

    u_flat = u.flatten()
    norm_u = np.linalg.norm(u_flat)
    if norm_u < 1e-8:
        return None
    u_hat = u_flat / norm_u

    best = 0.0
    for _freq, vec in phonons.imaginary_modes:
        v_flat = np.real(vec).flatten()
        norm_v = np.linalg.norm(v_flat)
        if norm_v < 1e-8:
            continue
        v_hat = v_flat / norm_v
        best = max(best, abs(float(np.dot(u_hat, v_hat))))
    return best
