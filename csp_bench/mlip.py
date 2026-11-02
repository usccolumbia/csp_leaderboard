"""MLIP backend abstraction.

Every energy/force call in the protocol — relaxation, ranking, ΔE, e_above_hull,
phonons — is required to go through *one frozen instance* of this interface
(see the protocol doc's "Frozen across the study" and "Ranker sensitivity"
sections). Functions elsewhere take an ``MLIP`` instance rather than
importing a specific package, so swapping MatterSim for a second ranker
(e.g. UMA, for the ranker-sensitivity control) never touches metric code.
"""
from __future__ import annotations

from dataclasses import dataclass

from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

_ADAPTOR = AseAtomsAdaptor()


@dataclass
class RelaxResult:
    structure: Structure
    energy: float
    forces: "list"
    energy_per_atom: float
    converged: bool
    n_steps: int


class MLIP:
    """Wraps an ASE ``Calculator``. Subclass only to override construction;
    all relax/single-point logic lives here so every backend behaves
    identically (same optimizer, same convergence test, same cell handling).
    """

    name: str = "generic"

    def __init__(self, calculator):
        self.calculator = calculator

    def _atoms(self, structure: Structure) -> Atoms:
        atoms = _ADAPTOR.get_atoms(structure)
        atoms.calc = self.calculator
        return atoms

    def single_point(self, structure: Structure) -> RelaxResult:
        atoms = self._atoms(structure)
        e = float(atoms.get_potential_energy())
        f = atoms.get_forces().tolist()
        return RelaxResult(structure, e, f, e / len(structure), True, 0)

    def relax(
        self,
        structure: Structure,
        fmax: float = 0.02,
        steps: int = 500,
        cell_relax: bool = True,
        constrain_symmetry: bool = False,
        logfile: str | None = None,
    ) -> RelaxResult:
        """Fixed protocol: fmax 0.02 eV/Å, 500-step cap, cell DOF on by
        default — matching "Frozen across the study". ``constrain_symmetry``
        wraps the atoms in ASE's ``FixSymmetry`` for the cascade's
        symmetry-constrained-relax stage (holds the predicted space group).
        """
        atoms = self._atoms(structure)
        constraints = []
        if constrain_symmetry:
            constraints.append(FixSymmetry(atoms))
        if constraints:
            atoms.set_constraint(constraints)

        target = FrechetCellFilter(atoms) if cell_relax else atoms
        dyn = BFGS(target, logfile=logfile or "/dev/null")
        converged = dyn.run(fmax=fmax, steps=steps)

        relaxed = _ADAPTOR.get_structure(atoms)
        e = float(atoms.get_potential_energy())
        f = atoms.get_forces().tolist()
        return RelaxResult(relaxed, e, f, e / len(atoms), bool(converged), dyn.get_number_of_steps())


class MatterSimMLIP(MLIP):
    """Chosen as the study's single frozen MLIP for phonon fidelity — see
    the protocol's resolved "which MLIP" open question."""

    name = "mattersim"

    def __init__(self, model: str = "MatterSim-v1.0.0-5M", device: str = "cpu"):
        from mattersim.forcefield import MatterSimCalculator

        super().__init__(MatterSimCalculator(load_path=model, device=device))


class CHGNetMLIP(MLIP):
    name = "chgnet"

    def __init__(self):
        from chgnet.model.dynamics import CHGNetCalculator

        super().__init__(CHGNetCalculator())


class MACEMLIP(MLIP):
    name = "mace-mp"

    def __init__(self, model: str = "medium", device: str = "cpu"):
        from mace.calculators import mace_mp

        super().__init__(mace_mp(model=model, device=device))


class UMAMLIP(MLIP):
    """Second, independent ranker for the ranker-sensitivity control.
    Requires ``fairchem`` (not installed in this environment yet) — install
    with ``pip install fairchem-core`` and a UMA checkpoint before use.
    """

    name = "uma"

    def __init__(self, model: str = "uma-s-1", device: str = "cpu"):
        try:
            from fairchem.core import pretrained_mlip
            from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        except ImportError as e:
            raise ImportError(
                "UMA backend needs `pip install fairchem-core` plus a downloaded "
                "checkpoint; see the ranker-sensitivity control in the protocol doc."
            ) from e

        predictor = pretrained_mlip.get_predict_unit(model, device=device)
        super().__init__(FAIRChemCalculator(predictor, task_name="omat"))


_BACKENDS = {
    "mattersim": MatterSimMLIP,
    "chgnet": CHGNetMLIP,
    "mace": MACEMLIP,
    "uma": UMAMLIP,
}


def get_mlip(name: str, **kwargs) -> MLIP:
    try:
        cls = _BACKENDS[name]
    except KeyError as e:
        raise ValueError(f"unknown MLIP backend {name!r}; choices: {sorted(_BACKENDS)}") from e
    return cls(**kwargs)
