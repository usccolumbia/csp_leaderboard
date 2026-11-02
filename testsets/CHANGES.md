# v3.1 (2026-10-08): single-target replacement in T46, T180 and T240

`mp-1183076` (Ac2AgPb) was replaced by `mp-3576` (HfGaCo2) in T46, T180 and T240 (one consistent replacement, as for the other shared small-tier targets).

Why: a re-check of the Materials Project `theoretical` flag found the reference purely theoretical while its DFT minimum phonon frequency is -1.60 THz, so it violated
the criterion 'phonon-stable or experimentally confirmed'. Replacement rules: same arity and category (ternary_easy), primitive cell of 4 sites (the structure type of the replaced
target: full Heusler, space group 225), e_hull <= 0.05 eV/atom, experimentally sourced MPTS-52 record (train.csv row 11560), identifier and composition not used elsewhere in the small tiers,
no structural match to any MP-20 train/validation structure (0 same-composition rows; checked at both matcher settings with and without supercell) and none to any current test structure.
Selected by lowest e_hull and lowest identifier among the 4-site space-group-225 ternary candidates passing these checks.
T800 is unchanged.
