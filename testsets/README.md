# Test sets (release v3.1, 8 October 2026)

| Set | Targets | Contents |
|---|---:|---|
| `T46/` | 46 | 33 targets shared with T180 |
| `T180/` | 180 | main test set: 60 binary, 60 ternary, 60 quaternary |
| `T240/` | 240 | T180 plus 60 five-element targets |
| `T800/` | 800 | MPTS-52 time-split test set |

Each set contains `cifs/<material_id>.cif` (reference structures), `queries.csv` / `queries.json` (the formula of the
primitive cell given to every method), `metadata.csv` and `records.json` (reference metadata),
`template_exclusions.json` (compositions a template method must exclude) and `manifest.json`.
`CIF_SHA256SUMS.json` lists the SHA-256 of every CIF and `CHANGES.md` records the v3.1 change.
Some reference CIFs are conventional cells; the query always gives the primitive cell.

## License and attribution

The reference structures and the metadata derived from them are a snapshot of
[Materials Project](https://materialsproject.org) data taken for this release (8 October 2026), redistributed under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/) of the Materials
Project. The repository's MIT license does not apply to them. The files are provided so that the benchmark can be
reproduced exactly; the Materials Project corrects and updates its data over time, so obtain current structures and
properties from the Materials Project, not from this snapshot.

Replacement targets and T800 were drawn from MPTS-52. If you use these test sets, please cite:

* A. Jain et al., "Commentary: The Materials Project: A materials genome approach to accelerating materials
  innovation", *APL Materials* **1**, 011002 (2013).
* S. G. Baird, H. M. Sayeed, J. Montoya and T. D. Sparks, "matbench-genmetrics: A Python library for benchmarking
  crystal structure generative models using time-based splits of Materials Project structures", *Journal of Open
  Source Software* **9**, 5618 (2024).
