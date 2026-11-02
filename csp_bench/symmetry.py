"""Axis A — symmetry ladder.

Five nested necessary conditions, each evaluated at a tolerance grid; the
gap between a tight and a loose symprec is itself diagnostic, because a
structure that is symmetric only at loose symprec is one relaxation will
usually fix.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import spglib
from pymatgen.core import Structure

SYMPREC_GRID = (0.01, 0.1, 0.5)


def crystal_system(spacegroup_number: int) -> str:
    n = spacegroup_number
    if 1 <= n <= 2:
        return "triclinic"
    if 3 <= n <= 15:
        return "monoclinic"
    if 16 <= n <= 74:
        return "orthorhombic"
    if 75 <= n <= 142:
        return "tetragonal"
    if 143 <= n <= 167:
        return "trigonal"
    if 168 <= n <= 194:
        return "hexagonal"
    if 195 <= n <= 230:
        return "cubic"
    raise ValueError(f"space group number out of range: {n}")


def _cell_tuple(structure: Structure):
    lattice = structure.lattice.matrix
    frac_coords = structure.frac_coords
    # species-index numbers double as the "decorated" channel: two sites with
    # different elements never collide in spglib's dataset.
    species = [str(sp) for sp in structure.species]
    unique = {s: i for i, s in enumerate(sorted(set(species)))}
    numbers = [unique[s] for s in species]
    return (lattice, frac_coords, numbers), species, unique


@dataclass
class SpglibDataset:
    number: int
    international: str
    wyckoffs: list[str]
    equivalent_atoms: np.ndarray
    species: list[str]
    raw: dict = field(repr=False, default_factory=dict)

    def orbit_letters(self, decorated: bool) -> Counter:
        """Multiset of (letter, multiplicity[, species]) — the Wyckoff
        "skeleton" (decorated=False) or the element-decorated version."""
        orbits: dict[int, list[int]] = {}
        for i, rep in enumerate(self.equivalent_atoms):
            orbits.setdefault(int(rep), []).append(i)
        counts = Counter()
        for rep, members in orbits.items():
            letter = self.wyckoffs[rep]
            mult = len(members)
            if decorated:
                sp = self.species[rep]
                counts[(letter, mult, sp)] += 1
            else:
                counts[(letter, mult)] += 1
        return counts


def spglib_dataset(structure: Structure, symprec: float) -> SpglibDataset:
    cell, species, _ = _cell_tuple(structure)
    raw = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if raw is None:
        raise ValueError(f"spglib failed to find symmetry at symprec={symprec}")
    return SpglibDataset(
        number=int(raw.number),
        international=str(raw.international),
        wyckoffs=list(raw.wyckoffs),
        equivalent_atoms=np.asarray(raw.equivalent_atoms),
        species=species,
        raw=dict(raw.__dict__) if hasattr(raw, "__dict__") else {},
    )


@dataclass
class SymmetryLadderResult:
    symprec: float
    system: bool
    sg: bool
    wyckoff: bool
    decorated: bool
    pred_sg_number: int
    gt_sg_number: int


def symmetry_ladder(pred: Structure, gt: Structure, symprec: float) -> SymmetryLadderResult:
    sp = spglib_dataset(pred, symprec)
    sg = spglib_dataset(gt, symprec)

    system_ok = crystal_system(sp.number) == crystal_system(sg.number)
    sg_ok = sp.number == sg.number  # raw number; non-standard settings handled below

    # (a) species-agnostic: does the site skeleton match?
    wyck_ok = sp.orbit_letters(decorated=False) == sg.orbit_letters(decorated=False)

    # (b) element-decorated: is the chemistry on the right sites?
    dec_ok = sp.orbit_letters(decorated=True) == sg.orbit_letters(decorated=True)

    return SymmetryLadderResult(
        symprec=symprec,
        system=system_ok,
        sg=sg_ok,
        wyckoff=wyck_ok,
        decorated=dec_ok,
        pred_sg_number=sp.number,
        gt_sg_number=sg.number,
    )


def symmetry_ladder_grid(pred: Structure, gt: Structure, symprecs=SYMPREC_GRID) -> dict[float, SymmetryLadderResult]:
    return {s: symmetry_ladder(pred, gt, s) for s in symprecs}


def symmetry_slack(pred: Structure, gt: Structure, tight: float = 0.01, loose: float = 0.5) -> int:
    """Loose-minus-tight space-group agreement.

    High slack (loose matches, tight doesn't) = approximately symmetric =
    strong prior that relaxation recovers the structure.
    """
    tight_ok = symmetry_ladder(pred, gt, tight).sg
    loose_ok = symmetry_ladder(pred, gt, loose).sg
    return int(loose_ok) - int(tight_ok)
