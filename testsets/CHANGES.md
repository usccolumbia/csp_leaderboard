# v3.1 (2026-10-08): single-target replacement in T46, T180 and T240

`mp-1183076` (Ac2AgPb) was replaced by `mp-3576` (HfGaCo2) in T46, T180 and T240 (one consistent replacement, as for the other shared small-tier targets).

Why: a re-check of the Materials Project `theoretical` flag found the reference purely theoretical while its DFT minimum phonon frequency is -1.60 THz, so it violated
the criterion 'phonon-stable or experimentally confirmed'. Replacement rules: same arity and category (ternary_easy), primitive cell of 4 sites (the structure type of the replaced
target: full Heusler, space group 225), e_hull <= 0.05 eV/atom, experimentally sourced MPTS-52 record (train.csv row 11560), identifier and composition not used elsewhere in the small tiers,
no structural match to any MP-20 train/validation structure (0 same-composition rows; checked at both matcher settings with and without supercell) and none to any current test structure.
Selected by lowest e_hull and lowest identifier among the 4-site space-group-225 ternary candidates passing these checks.
T800 is unchanged.

# Correction (1 November 2026): query formula of mp-557387 in T180 and T240

The v3.1 release gave mp-557387 (Sr2TmNbO6) the 10-atom query `Nb1O6Sr2Tm1` and marked it `submission_allowed=False`
(a StructureMatcher reduction flagged for symmetry review). Its primitive cell at a tolerance of 0.01 (pymatgen and
spglib) has 20 atoms, and every method on the leaderboard was run on `Sr4Tm2Nb2O12`. The query, `metadata.csv` and
`records.json` now give `Sr4Tm2Nb2O12` (20 sites) and the target is open for submission. The reference CIF and its hash
are unchanged.
