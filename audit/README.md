# MP-20 leakage audit and replacement records

Every target of every test set was re-read from its CIF file and compared with the 36,183 structures of the MP-20
training and validation splits. Compatible pairs (same elemental fractions, from the parsed CIFs) were tested in both
directions with pymatgen `StructureMatcher` (`primitive_cell=True`, `scale=True`, `allow_subset=False`) at the
evaluation tolerances (`ltol` 0.2, `stol` 0.3, `angle_tol` 5) and at looser tolerances (0.3, 0.5, 10), each with
`attempt_supercell` False and True: eight checks per pair. Matching targets were replaced.

* `replacements.csv`: every replaced target (test set, old and new identifier, formula, CIF hashes, reason and MPTS-52
  source record). Rows: T46 7, T180 36, T240 42, T800 34. Replacements for T46, T180 and T240 were drawn from
  the MPTS-52 train, validation and test records; replacements for T800 only from the MPTS-52 test split. The last
  three rows record the v3.1 reference-quality replacement (mp-1183076 by mp-3576; see `../testsets/CHANGES.md`).
* `mp20_pairwise_checks.csv`: every compatible test/MP-20 pair of the released sets with the result of each of the
  eight checks (columns `standard_*`: evaluation tolerances, `project_*`: loose tolerances; `supercell`: with supercell
  matching). All 190 pairs fail all eight checks. Test structures are identified by the SHA-256 of their CIF
  (see `../testsets/CIF_SHA256SUMS.json`).
