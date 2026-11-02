"""Fig. 4 of the paper: where methods break. Top-1 success by primitive-cell size, by reference energy above
the hull and by prototype availability, for the template baseline, the leading generators and the older/specialised models."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; R = HERE / "data"; OUT = HERE / "output" / "paper_figures"
t = pd.read_csv(R / "t180_topk_sg_per_case_all.csv"); md = pd.read_csv(HERE.parent / "testsets/T180/metadata.csv").set_index("material_id")
mp20 = json.loads((R / "mp20_prototype_counts.json").read_text())
LEAD = ["crystalflow", "crystalite", "omatg", "uni3dar", "equicsp", "uflowcsp", "mattergen", "diffcsp"]
OLD = ["cond-cdvae", "gemsdiff", "tgdmat", "cdvae", "symmcd", "crystalgrw", "crystalformer"]
GROUPS = [("Template (TCSPv3)", ["tcspv3"], "#2a78d6"), ("Eight leading generators", LEAD, "#e69f00"), ("Seven older / specialised models", OLD, "#7a6fb0")]
ids = sorted(md.index); t["hit"] = (t.sm_rank <= 1).fillna(False).astype(bool)
H = {a: t[t.algorithm == a].set_index("material_id").hit.reindex(ids).fillna(False).to_numpy() for a in t.algorithm.unique()}
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return p, c - h, c + h
def curve(mask_list):
    out = {}
    for name, algs, _ in GROUPS:
        res = []
        for m in mask_list:
            k = sum(int(H[a][m].sum()) for a in algs); n = len(algs) * int(m.sum()); res.append(wilson(k, n) + (int(m.sum()),))
        out[name] = res
    return out
ns = md.primitive_nsites.reindex(ids).to_numpy(); eh = md.dft_e_above_hull.reindex(ids).to_numpy()
size_bins = [(2, 4), (5, 8), (9, 12), (13, 20), (21, 28)]; size_lab = ["2–4", "5–8", "9–12", "13–20", "21–28"]
eh_bins = [(-1, 1e-6), (1e-6, .025), (.025, .05), (.05, .1)]; eh_lab = ["on hull", "0–25", "25–50", "50–100"]
proto = np.array([mp20[i]["n_same_sg"] > 0 for i in ids])
size_m = [(ns >= a) & (ns <= b) for a, b in size_bins]; eh_m = [(eh > a) & (eh <= b) for a, b in eh_bins]; pr_m = [proto, ~proto]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1, .62], wspace=.38))
panels = [("a", size_m, size_lab, "Primitive-cell size (atoms)", axs[0]), ("b", eh_m, eh_lab, "Reference energy above hull (meV/atom)", axs[1]), ("c", pr_m, ["yes", "no"], "MP-20 prototype available", axs[2])]
tab = {}
for letter, masks, labs, xl, ax in panels:
    cv = curve(masks); x = np.arange(len(masks))
    for gi, (name, _, col) in enumerate(GROUPS):
        p = np.array([r[0] for r in cv[name]]) * 100; lo = np.array([r[1] for r in cv[name]]) * 100; hi = np.array([r[2] for r in cv[name]]) * 100
        off = (gi - 1) * 0.0
        if letter == "c":
            ax.bar(x + (gi - 1) * .26, p, .24, color=col, label=name); ax.errorbar(x + (gi - 1) * .26, p, yerr=[p - lo, hi - p], fmt="none", ecolor="#444", lw=.8, capsize=1.5)
        else:
            ax.plot(x, p, "-o", color=col, lw=1.6, ms=4, label=name); ax.fill_between(x, lo, hi, color=col, alpha=.15, lw=0)
        tab[(letter, name)] = [(l, round(float(a), 1), r[3]) for l, a, r in zip(labs, p, cv[name])]
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={int(m.sum())})" for l, m in zip(labs, masks)] if letter != "c" else [f"{l}\n(n={int(m.sum())})" for l, m in zip(labs, masks)], fontsize=6.6)
    ax.set_xlabel(xl, fontsize=7.4); ax.set_ylim(0, 100); ax.grid(axis="y", color="#e4e4e4", lw=.6); ax.set_axisbelow(True)
    ax.text(-0.2 if letter != "c" else -0.34, 1.07, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
axs[0].set_ylabel("Top-1 match rate (%)"); h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(.5, -.04))
fig.subplots_adjust(bottom=.3, left=.07, right=.99, top=.9)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_failure_regimes.{ext}", dpi=300, bbox_inches="tight")
for k, v in tab.items(): print(k, v)
import json as _j; (OUT / "fig_failure_regimes_values.json").write_text(_j.dumps({f"{k[0]}|{k[1]}": v for k, v in tab.items()}, indent=1))
