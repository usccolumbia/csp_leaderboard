"""Post-processing cascade and the F0-F9 failure taxonomy.

The cascade is applied in a fixed order with a match check after every
stage; the stage at which a prediction first matches determines its
bucket. Buckets are mutually exclusive and exhaustive by construction (the
first matching branch wins), which is what makes the stacked-bar taxonomy
figure a true decomposition rather than a set of overlapping indicators.

    F0   matched as generated
    F1   symmetry-slack recoverable (loose symmetrisation alone)
    F2   lattice-scale recoverable (volume rescale alone)
    F3a  symmetry-constrained-relax recoverable (predicted SG held)
    F3b  free-relax recoverable (SG allowed to drift)
    F8   pred relaxes BELOW the reference — a better polymorph, not a failure
    F4   correct decorated Wyckoff, wrong basin (unrecoverable)
    F5   correct space group, wrong Wyckoff decoration
    F6   correct crystal system or topology, wrong space group
    F7   correct local chemistry (CN), wrong assembly
    F9   wrong local chemistry
"""
from __future__ import annotations

from dataclasses import dataclass

import spglib
from pymatgen.core import Structure

from .bonding import BondingMetrics, bonding_metrics
from .energy import EnergyMetrics, energy_metrics
from .matcher import FROZEN_MATCHER, match
from .mlip import MLIP, RelaxResult
from .structure_utils import rescale_to_volume
from .symmetry import SymmetryLadderResult, symmetry_ladder

BUCKET_ORDER = ["F0", "F1", "F2", "F3a", "F3b", "F8", "F4", "F5", "F6", "F7", "F9"]


def symmetrize(structure: Structure, symprec: float = 0.5) -> Structure:
    """Loose-symprec spglib standardization: snaps near-symmetric
    coordinates onto the symmetry they almost already have, without
    touching energetics via a relaxation."""
    lattice = structure.lattice.matrix
    positions = structure.frac_coords
    # spglib needs integer type labels, not pymatgen Species objects
    unique = {str(s): i for i, s in enumerate(sorted({str(s) for s in structure.species}))}
    numbers = [unique[str(s)] for s in structure.species]

    cell = (lattice, positions, numbers)
    std = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False, symprec=symprec)
    if std is None:
        return structure
    std_lattice, std_positions, std_numbers = std
    inv_unique = {i: sp for sp, i in unique.items()}
    species = [inv_unique[n] for n in std_numbers]
    return Structure(std_lattice, species, std_positions)


def constrained_relax(structure: Structure, mlip: MLIP, fmax: float = 0.02) -> RelaxResult:
    """Free relaxation is not what stage 4 means: holding the predicted
    space group (ASE's ``FixSymmetry``) is what cleanly separates
    "symmetry right, coordinates wrong" from "symmetry wrong", which a
    fully free relaxation blurs by letting the group change mid-trajectory.
    """
    return mlip.relax(structure, fmax=fmax, cell_relax=True, constrain_symmetry=True)


def free_relax(structure: Structure, mlip: MLIP, fmax: float = 0.02) -> RelaxResult:
    return mlip.relax(structure, fmax=fmax, cell_relax=True, constrain_symmetry=False)


@dataclass
class CascadeTrace:
    bucket: str
    stage_matches: dict[str, bool]
    relax_rmsd: float | None      # how far stage 4/5 moved to win the match
    symmetry: SymmetryLadderResult | None
    bonding: BondingMetrics | None
    energy: EnergyMetrics | None


def assign_bucket(
    pred: Structure,
    gt: Structure,
    mlip: MLIP,
    matcher=None,
    symprec_ladder: float = 0.1,
    cn_threshold: float = 0.9,
) -> CascadeTrace:
    matcher = matcher or FROZEN_MATCHER
    stage_matches: dict[str, bool] = {}

    if match(pred, gt, matcher):
        stage_matches["F0"] = True
        return CascadeTrace("F0", stage_matches, 0.0, None, None, None)

    symmetrised = symmetrize(pred)
    if match(symmetrised, gt, matcher):
        stage_matches["F1"] = True
        return CascadeTrace("F1", stage_matches, None, None, None, None)

    rescaled = rescale_to_volume(pred, gt.volume)
    if match(rescaled, gt, matcher):
        stage_matches["F2"] = True
        return CascadeTrace("F2", stage_matches, None, None, None, None)

    c_relax = constrained_relax(pred, mlip)
    rmsd = matcher.get_rms_dist(c_relax.structure, gt)
    if match(c_relax.structure, gt, matcher):
        stage_matches["F3a"] = True
        return CascadeTrace("F3a", stage_matches, rmsd[0] if rmsd else None, None, None, None)

    f_relax = free_relax(pred, mlip)
    rmsd = matcher.get_rms_dist(f_relax.structure, gt)
    if match(f_relax.structure, gt, matcher):
        stage_matches["F3b"] = True
        return CascadeTrace("F3b", stage_matches, rmsd[0] if rmsd else None, None, None, None)

    e = energy_metrics(pred, gt, mlip)
    if e.found_lower:
        stage_matches["F8"] = True
        return CascadeTrace("F8", stage_matches, None, None, None, e)

    a = symmetry_ladder(pred, gt, symprec=symprec_ladder)
    b = bonding_metrics(pred, gt, normalise_volume=True)

    if a.decorated:
        bucket = "F4"
    elif a.sg:
        bucket = "F5"
    elif a.system or b.topology_match:
        bucket = "F6"
    elif b.cn_accuracy is not None and b.cn_accuracy > cn_threshold:
        bucket = "F7"
    else:
        bucket = "F9"

    stage_matches[bucket] = True
    return CascadeTrace(bucket, stage_matches, None, a, b, e)


def bucket_rank(bucket: str) -> int:
    """Ordering used to combine bucket labels across k candidates
    (``min`` over k in bucket rank = "best outcome any sample reached")."""
    return BUCKET_ORDER.index(bucket)
