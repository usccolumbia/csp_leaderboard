# Analysis

Scripts that reproduce the T180 figures, tables and quoted numbers of the paper from the per-target tables in
[`data/`](data). They need `numpy`, `pandas`, `scipy`, `matplotlib` and `pymatgen`, and write to `output/`.

```bash
cd analysis
python figure_failure_regimes.py      # Fig. 4 (run before numbers_t180.py)
python figures_t180_diagnosis.py      # Figs. 2-3, the supplementary diagnosis figures and the method cards
python tables_t180.py                 # Table 2 and the pairwise and prototype-availability tables
python tables_t180_supplementary.py   # Supplementary tables: coverage, space-group tolerance, arity, generator-only targets
python numbers_t180.py                # numbers quoted in the text: paired tests, complementarity, prototype availability
```

| Script | Output |
|---|---|
| `figure_failure_regimes.py` | `output/paper_figures/fig_failure_regimes.*`: top-1 rate by cell size, energy above hull and prototype availability |
| `figures_t180_diagnosis.py` | `output/paper_figures/`: `fig_overall_performance`, `fig_bucket_composition`, `fig_technique_dendrogram`, `fig_cluster_profile`, `fig_diagnostic_heatmap`; `output/diagnosis_figures/cards/`: one diagnostic card per method |
| `technique_clustering.py` | technique feature matrix and clustering, imported by `figures_t180_diagnosis.py` |
| `tables_t180.py` | `output/tables/t180_main_table.csv`, `t180_decomp_*.csv`, `t180_prototype_availability.csv` |
| `tables_t180_supplementary.py` | `output/tables/supp_tables.json` (LaTeX rows) |
| `numbers_t180.py` | printed to the terminal |

## Data

| File | Contents |
|---|---|
| `t180_topk_sg_per_case_all.csv` | per method and target: number of valid candidates, rank of the first match, top-1 space group of candidate and reference at `symprec` 0.1 and 0.01 |
| `raw_top1_cascade.csv` | per method and target: F0–F9 diagnosis of the raw (unrelaxed) top-1 candidate with the axis metrics behind it, computed with [`csp_bench`](../csp_bench) |
| `t180_all_algorithms_summary.csv` | per method: summary of the diagnosis |
| `bootstrap_ci_topk.csv` | per method: top-1/5/20 with 95% bootstrap intervals (reproduced by `metrics/compare.py`) |
| `t180_strict_primitive_per_case.csv` | per method and target: rank of the first match when both structures are reduced to primitive cells and matched without `primitive_cell` (sensitivity check) |
| `mp20_prototype_counts.json`, `library_prototype_counts.json` | per target: number of MP-20 training/validation structures and TCSPv3 library structures with the same stoichiometry and space group |

The tables were produced from the 20 ranked candidates of every method with the evaluator in [`metrics/`](../metrics)
and the diagnosis package [`csp_bench`](../csp_bench). TCSPv3 is relaxed and ranked with ORB-v3 like the generators.
The framework flowchart (Fig. 1) and the F0–F9 cascade diagram are drawings and have no script.
