"""Results figures of the diagnosis paper for the 16 algorithms with T180 results.

Reads  data/{t180_topk_sg_per_case_all.csv, t180_all_algorithms_summary.csv, raw_top1_cascade.csv, bootstrap_ci_topk.csv}
Writes output/diagnosis_figures/:
  fig_technique_dendrogram   technique dendrogram + feature matrix (16 algorithms)
  fig_overall_performance    (a) cumulative match rate vs k, (b) top-1 / top-20 dumbbell
  fig_bucket_composition     (a) F0-F9 composition of the RAW top-1, (b) relaxation lift
  fig_cluster_profile        technique-cluster level profile
  fig_diagnostic_heatmap     algorithm x diagnostic metric heatmap in dendrogram order
  cards/card_<algo>          one-page diagnostic profile per algorithm; supp_algorithm_cards.pdf
Technique families and the feature matrix are those of technique_clustering.py.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, fcluster

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import technique_clustering as tc  # noqa: E402

RIN = HERE / "data"
OUT = HERE / "output" / "diagnosis_figures"; (OUT / "cards").mkdir(parents=True, exist_ok=True)
PAPER = HERE / "output" / "paper_figures"   # the figures used in the paper
RCSV = OUT
INK, INK2, GRID = tc.INK, tc.INK2, tc.GRID
KEY = {"TCSPv3": "tcspv3", "CrystalFlow": "crystalflow", "Crystalite": "crystalite", "OMatG": "omatg", "Uni-3DAR": "uni3dar",
       "uFlowCSP": "uflowcsp", "EquiCSP": "equicsp", "MatterGen": "mattergen", "DiffCSP": "diffcsp", "cond-CDVAE": "cond-cdvae",
       "GemsDiff": "gemsdiff", "TGDMat": "tgdmat", "SymmCD": "symmcd", "CrystalGRW": "crystalgrw", "CrystalFormer": "crystalformer", "CDVAE": "cdvae"}
FAM = tc.FAMILIES
# bucket groups for the stacked bars (ordered by severity); Okabe-Ito colors
GROUPS = [("F0  matched as generated", ["F0"], "#0072B2"),
          ("F1-F3  fixed by symmetrize / rescale / relax", ["F1", "F2", "F3a", "F3b"], "#56B4E9"),
          ("F8  relaxes below the reference", ["F8"], "#009E73"),
          ("F4-F5  right symmetry family, wrong structure", ["F4", "F5"], "#E69F00"),
          ("F6-F7  right crystal system or local chemistry", ["F6", "F7"], "#CC79A7"),
          ("F9  wrong local chemistry", ["F9"], "#D55E00"),
          ("no valid candidate", ["none"], "#bdbdb6")]


def style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})


def save(fig, stem, paper=False):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
        if paper:
            PAPER.mkdir(exist_ok=True); shutil.copy(OUT / f"{stem}.{ext}", PAPER / f"{stem}.{ext}")
    plt.close(fig)


def load():
    df = tc.build_matrix(); df = df[df.algorithm.isin(KEY)].reset_index(drop=True)
    Z, _ = tc.cluster(df)
    leaves = dendrogram(Z, no_plot=True)["leaves"]; order = [df.algorithm[i] for i in leaves]
    topk = pd.read_csv(RIN / "t180_topk_sg_per_case_all.csv")
    inv = {v: k for k, v in KEY.items()}; topk["name"] = topk.algorithm.map(inv)
    topk = topk[topk.name.notna()]
    summ = pd.read_csv(RIN / "t180_all_algorithms_summary.csv").set_index("algo")
    raw = pd.read_csv(RIN / "raw_top1_cascade.csv"); raw["name"] = raw.algorithm.map(inv); raw = raw[raw.name.notna()]
    fam = df.set_index("algorithm").family_index.to_dict()
    return df, Z, order, topk, summ, raw, fam


def fam_marker(ax, x, y, f, size=7):
    _, c, m = FAM[f]; ax.plot(x, y, marker=m, color=c, markersize=size, markeredgecolor="white", clip_on=False, linestyle="")


def family_legend(fig_or_ax, used, **kw):
    h = [Line2D([], [], marker=m, linestyle="", markerfacecolor=c, markeredgecolor="white", markersize=8, label=l)
         for i, (l, c, m) in enumerate(FAM) if i in used]
    return fig_or_ax.legend(handles=h, frameon=False, fontsize=8, title="Primary technique", title_fontsize=8, **kw)


def spread(ys, gap):
    idx = np.argsort(ys); out = np.array(ys, float)
    for a, b in zip(idx[:-1], idx[1:]):
        if out[b] - out[a] < gap: out[b] = out[a] + gap
    return out


def rate_curve(g, n=180):  # n = number of targets of the test set
    r = g.sm_rank.to_numpy(); return np.array([np.nansum(r <= k) / n * 100 for k in range(1, 21)])


# ------------------------------------------------------------------ figures
def fig_dendrogram(df, Z):
    tc.OUT = OUT  # the module's save helper writes there
    tc.fig_clustermap(df, Z)
    for ext in ("pdf", "png"):
        shutil.move(OUT / f"technique_clustermap.{ext}", OUT / f"fig_technique_dendrogram.{ext}")
        PAPER.mkdir(exist_ok=True); shutil.copy(OUT / f"fig_technique_dendrogram.{ext}", PAPER / f"fig_technique_dendrogram.{ext}")


def fig_overall(order, topk, summ, fam):
    CI = pd.read_csv(RIN / "bootstrap_ci_topk.csv")
    fig = plt.figure(figsize=(11.4, 5.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.55)
    ax = fig.add_subplot(gs[0]); ks = np.arange(1, 21); ends = {}
    for name in order:
        g = topk[topk.name == name]; y = rate_curve(g); f = fam[name]; _, c, m = FAM[f]
        lw, z = (3.0, 5) if name == "TCSPv3" else (1.4, 3)
        ax.plot(ks, y, color=c, lw=lw, marker=m, markersize=3.6 if name != "TCSPv3" else 5, markevery=[0, 4, 19], zorder=z, alpha=1.0 if name == "TCSPv3" else 0.9)
        ends[name] = y[-1]
    names = list(ends); ys = spread([ends[n] for n in names], 3.6)
    for n, y in zip(names, ys):
        c = FAM[fam[n]][1]
        ax.plot([20.0, 20.9, 21.5], [ends[n], y, y], color=c, lw=0.8, clip_on=False, solid_capstyle="round")
        ax.text(21.7, y, n, va="center", fontsize=7.8, color=INK, fontweight="bold" if n == "TCSPv3" else "normal", clip_on=False)
    ax.set_xlim(1, 20); ax.set_ylim(0, 100); ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlabel("Number of candidates considered, k")
    ax.set_ylabel("Targets matched within top-k (%, of 180)"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.text(-0.14, 1.03, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")
    # dumbbell
    ax2 = fig.add_subplot(gs[1]); t = summ.loc[[n for n in order if n in summ.index]].sort_values("top1", ascending=True)
    for i, (name, r) in enumerate(t.iterrows()):
        f = fam[name]; _, c, m = FAM[f]
        ax2.hlines(i, r["top1%"], r["top20%"], color=GRID, lw=3, zorder=1)
        for k_, mk_ in ((1, 0.17), (20, -0.17)):
            q = CI[(CI.algo == name) & (CI.k == k_)].iloc[0]
            ax2.hlines(i + mk_, q.lo, q.hi, color=c, lw=1.0, alpha=0.75, zorder=2)
            ax2.vlines([q.lo, q.hi], i + mk_ - 0.07, i + mk_ + 0.07, color=c, lw=1.0, alpha=0.75, zorder=2)
        ax2.scatter(r["top1%"], i, s=46, color=c, marker=m, edgecolor="white", zorder=3)
        ax2.scatter(r["top20%"], i, s=46, facecolor="white", edgecolor=c, marker=m, linewidth=1.5, zorder=3)
        ax2.text(CI[(CI.algo == name) & (CI.k == 20)].iloc[0].hi + 1.5, i, f"{int(r.top1)} / {int(r.top20)}", va="center", fontsize=7.6, color=INK2)
    ax2.set_yticks(range(len(t))); ax2.set_yticklabels(t.index, fontsize=8.4)
    for tl in ax2.get_yticklabels():
        if tl.get_text() == "TCSPv3": tl.set_fontweight("bold")
    ax2.set_xlim(0, 122); ax2.set_xticks([0, 20, 40, 60, 80, 100]); ax2.set_xlabel("Match rate (%): top-1 (filled), top-20 (open), 95% CI"); ax2.grid(axis="x", color=GRID, lw=0.8); ax2.set_axisbelow(True)
    ax2.text(-0.30, 1.03, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold")
    family_legend(fig, set(fam.values()), loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=6, columnspacing=1.2, handletextpad=0.3)
    save(fig, "fig_overall_performance", paper=True)


def bucket_table(raw, order):
    rows = {}
    for name in order:
        g = raw[raw.name == name]; n = len(g); vc = g.bucket_primary.value_counts()
        row = {lab: sum(int(vc.get(b, 0)) for b in bs if b != "none") for lab, bs, _ in GROUPS}
        row["no valid candidate"] = 180 - int(g.bucket_primary.notna().sum())
        rows[name] = row
    return pd.DataFrame(rows).T


def fig_buckets(order, raw, summ, fam):
    B = bucket_table(raw, order); fig = plt.figure(figsize=(11.6, 5.8)); gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 0.75], wspace=0.06)
    ax = fig.add_subplot(gs[0]); n = len(order)
    for i, name in enumerate(order):
        left = 0
        for lab, _, col in GROUPS:
            v = B.loc[name, lab]
            if v: ax.barh(i, v / 180 * 100, left=left, color=col, edgecolor="white", linewidth=0.6, height=0.78)
            if v >= 9: ax.text(left + v / 180 * 50, i, str(int(v)), ha="center", va="center", fontsize=7, color="white" if col in ("#0072B2", "#D55E00", "#009E73") else INK)
            left += v / 180 * 100
    ax.set_yticks(range(n)); ax.set_yticklabels(order); ax.tick_params(axis="y", pad=16); ax.set_ylim(n - 0.5, -0.5); ax.set_xlim(0, 100)
    for i, name in enumerate(order):
        fam_marker(ax, -2.2, i, fam[name], 6.5)
    for tl in ax.get_yticklabels():
        if tl.get_text() == "TCSPv3": tl.set_fontweight("bold")
    # separators between technique clusters (consecutive same-family leaves)
    for i in range(n - 1):
        if fam[order[i]] != fam[order[i + 1]]: ax.axhline(i + 0.5, color=INK2, lw=0.6, ls=(0, (3, 3)))
    ax.set_xlabel("Share of the 180 targets (%): outcome of the raw (unrelaxed) top-1 candidate"); ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.text(-0.17, 1.02, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for l, _, c in GROUPS], loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False, fontsize=7.8)
    # (b) relaxation lift = exported - raw top-1 (targets)
    ax2 = fig.add_subplot(gs[1], sharey=ax); lift = []
    for name in order:
        raw_m = int((raw[raw.name == name].match_primary_raw == True).sum()); lift.append(int(summ.loc[name, "top1"]) - raw_m)
    ax2.barh(range(n), lift, color="#6e6d68", height=0.62)
    for i, v in enumerate(lift): ax2.text(v + 1.2, i, str(v), va="center", fontsize=7.6, color=INK2)
    ax2.set_xlim(0, max(lift) * 1.2); ax2.tick_params(axis="y", left=False, labelleft=False); ax2.set_xlabel("Extra top-1 targets matched after\nORB relaxation of the same candidate")
    ax2.grid(axis="x", color=GRID, lw=0.7); ax2.set_axisbelow(True); ax2.text(-0.04, 1.02, "b", transform=ax2.transAxes, fontsize=12, fontweight="bold")
    save(fig, "fig_bucket_composition", paper=True); B.to_csv(RCSV / "diagnosis_bucket_groups.csv")


def fig_cluster_profile(order, raw, summ, fam):
    rows = []
    for f in sorted(set(fam.values())):
        members = [n for n in order if fam[n] == f]
        g = raw[raw.name.isin(members)]; vc = g.bucket_primary.value_counts(); tot = int(g.bucket_primary.notna().sum())
        row = dict(family=FAM[f][0], n=len(members), members=", ".join(members), top1=np.median([summ.loc[m, "top1%"] for m in members]),
                   top20=np.median([summ.loc[m, "top20%"] for m in members]), gap=np.median([summ.loc[m, "gap"] for m in members]))
        for lab, bs, _ in GROUPS[:-1]: row[lab] = sum(int(vc.get(b, 0)) for b in bs) / max(tot, 1) * 100
        rows.append(row)
    C = pd.DataFrame(rows).sort_values("top1", ascending=False); C.to_csv(RCSV / "diagnosis_cluster_profile.csv", index=False)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.05}, sharey=True)
    y = np.arange(len(C))
    for i, (_, r) in enumerate(C.iterrows()):
        left = 0
        for lab, _, col in GROUPS[:-1]:
            ax.barh(i, r[lab], left=left, color=col, edgecolor="white", linewidth=0.6, height=0.7)
            if r[lab] >= 7: ax.text(left + r[lab] / 2, i, f"{r[lab]:.0f}", ha="center", va="center", fontsize=7.4, color="white" if col in ("#0072B2", "#D55E00", "#009E73") else INK)
            left += r[lab]
    ax.set_yticks(y); ax.set_yticklabels([f"{r.family}  (n={r.n})" for _, r in C.iterrows()]); ax.tick_params(axis="y", pad=16); ax.set_ylim(len(C) - 0.5, -0.5); ax.set_xlim(0, 100)
    ax.set_xlabel("Pooled outcome of the raw top-1 candidate (% of diagnosed targets)"); ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    fam_idx = {FAM[f][0]: f for f in set(fam.values())}
    for i, (_, r) in enumerate(C.iterrows()): fam_marker(ax, -1.6, i, fam_idx[r.family], 6.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for l, _, c in GROUPS[:-1]], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=7.6)
    ax2.hlines(y, C.top1, C.top20, color=GRID, lw=3)
    ax2.scatter(C.top1, y, s=50, color=INK, zorder=3); ax2.scatter(C.top20, y, s=50, facecolor="white", edgecolor=INK, linewidth=1.5, zorder=3)
    for i, (_, r) in enumerate(C.iterrows()): ax2.text(max(r.top20, r.top1) + 2.5, i, f"{r.top1:.0f} / {r.top20:.0f}", va="center", fontsize=7.6, color=INK2)
    ax2.set_xlim(0, 108); ax2.set_xlabel("Median top-1 (filled) and top-20 (open) match rate (%)"); ax2.grid(axis="x", color=GRID, lw=0.7); ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", left=False)
    save(fig, "fig_cluster_profile", paper=True)


def fig_heatmap(order, raw, summ, topk, fam):
    cols = []
    def col(name, f, fmt="{:.0f}", better_high=True): cols.append((name, f, fmt, better_high))
    sg_m = lambda n: raw[(raw.name == n) & (raw.match_primary_raw == True)].sg_ok.astype(float).mean() * 100
    un = lambda n, c, fn: fn(raw[(raw.name == n) & (raw.match_primary_raw == False)][c].astype(float))
    col("Raw top-1", lambda n: 100 * (raw[raw.name == n].match_primary_raw == True).sum() / 180)
    col("Exported top-1", lambda n: summ.loc[n, "top1%"])
    col("Top-20", lambda n: summ.loc[n, "top20%"])
    col("Ranking headroom:\ntop-20 minus top-1 (points)", lambda n: summ.loc[n, "gap"] / 180 * 100, better_high=None)
    col("Space group of matched\nraw top-1 (%)", sg_m)
    col("Space group of exported\ntop-1, symprec 0.1 (%)", lambda n: summ.loc[n, "sg%"])
    col("F8: relaxes below\nreference (%)", lambda n: 100 * (raw[raw.name == n].bucket_primary == "F8").sum() / 180, better_high=None)
    col("Median |eps_vol| of\nunmatched", lambda n: un(n, "eps_vol", lambda s: s.abs().median()), "{:.2f}", False)
    col("Median eps_shape of\nunmatched", lambda n: un(n, "eps_shape", lambda s: s.median()), "{:.2f}", False)
    col("Median |delta CDM| of\nunmatched", lambda n: un(n, "delta_cdm", lambda s: s.abs().median()), "{:.3f}", False)
    col("Median CN accuracy of\nunmatched", lambda n: un(n, "cn_accuracy", lambda s: s.median()), "{:.2f}", True)
    V = np.array([[f(n) for _, f, _, _ in cols] for n in order], float)
    Zs = (V - np.nanmean(V, 0)) / np.nanstd(V, 0)
    for j, (_, _, _, bh) in enumerate(cols):
        if bh is None: Zs[:, j] = 0.0            # context column, not scored
        elif not bh: Zs[:, j] = -Zs[:, j]        # orient so that high = favourable
    cmap = LinearSegmentedColormap.from_list("div", ["#c8553d", "#f1efe9", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    ax.imshow(Zs, cmap=cmap, vmin=-2.2, vmax=2.2, aspect="auto")
    for i in range(len(order)):
        for j, (_, _, fmt, _) in enumerate(cols):
            if not np.isnan(V[i, j]): ax.text(j, i, fmt.format(V[i, j]), ha="center", va="center", fontsize=7.8, color=INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], rotation=50, ha="left", fontsize=8, rotation_mode="anchor"); ax.xaxis.tick_top()
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xticks(np.arange(len(cols)) - 0.5, minor=True); ax.set_yticks(np.arange(len(order)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.6); ax.tick_params(which="both", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for i, name in enumerate(order): fam_marker(ax, -0.62, i, fam[name], 6.5)
    for i in range(len(order) - 1):
        if fam[order[i]] != fam[order[i + 1]]: ax.axhline(i + 0.5, color=INK, lw=1.0)
    ax.axvline(3.5, color=INK, lw=1.0); ax.axvline(6.5, color=INK, lw=1.0)
    ax.annotate("Column colour: z-score within the column, oriented so that blue = more favourable (higher match, symmetry and chemistry; smaller strain). Grey columns are context and not scored.",
                xy=(0, -0.02), xycoords="axes fraction", va="top", fontsize=7.6, color=INK2)
    save(fig, "fig_diagnostic_heatmap", paper=True)
    pd.DataFrame(V, index=order, columns=[c[0].replace("\n", " ") for c in cols]).to_csv(RCSV / "diagnosis_heatmap_values.csv")


def card(name, fam, raw, topk, summ, n_total=180, baseline="TCSPv3"):
    g = raw[raw.name == name]; gd = g[g.bucket_primary.notna()]; fi = fam[name]; _, color, _ = FAM[fi]
    fig, axs = plt.subplots(2, 4, figsize=(13.0, 6.2)); axs = axs.ravel()
    n = len(gd); matched = gd.match_primary_raw == True
    fig.suptitle(f"{name} -- {FAM[fi][0]}   |   diagnosed targets: {n} of {n_total}   |   raw top-1 {int(matched.sum())}, exported top-1 {int(summ.loc[name, 'top1'])}, top-20 {int(summ.loc[name, 'top20'])}", fontsize=10.5, x=0.01, ha="left")
    # a: bucket bars
    ax = axs[0]; vc = gd.bucket_primary.value_counts(); labs = ["F0", "F1", "F2", "F3a", "F3b", "F8", "F4", "F5", "F6", "F7", "F9"]
    gcol = {b: c for _, bs, c in GROUPS for b in bs}
    ax.barh(range(len(labs)), [int(vc.get(b, 0)) for b in labs], color=[gcol[b] for b in labs], height=0.7)
    for i, b in enumerate(labs): ax.text(int(vc.get(b, 0)) + 0.8, i, str(int(vc.get(b, 0))), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.invert_yaxis(); ax.set_xlabel("Targets (raw top-1, campaign matcher)"); ax.set_title("a  Failure-mode bucket", loc="left", fontsize=9)
    # b: success vs k with TCSPv3
    ax = axs[1]; ks = np.arange(1, 21)
    ax.plot(ks, rate_curve(topk[topk.name == baseline], n_total), color="#6e6d68", lw=2, ls=(0, (4, 2)), label=baseline)
    ax.plot(ks, rate_curve(topk[topk.name == name], n_total), color=color, lw=2.2, label=name); ax.set_ylim(0, 100); ax.set_xlim(1, 20); ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("k"); ax.set_ylabel("matched within top-k (%)"); ax.legend(frameon=False, fontsize=8); ax.set_title("b  Match rate vs candidates", loc="left", fontsize=9)
    # c: symmetry ladder
    ax = axs[2]; steps = [("crystal\nsystem", "sys_ok"), ("space\ngroup", "sg_ok"), ("Wyckoff\nsites", "wyckoff_ok"), ("decorated\nWyckoff", "decorated_ok")]
    for k, (lab, c) in enumerate(steps):
        for off, sel, col_ in ((-0.19, matched, "#0072B2"), (0.19, ~matched, "#D55E00")):
            v = gd[sel][c].astype(float).mean() * 100 if sel.sum() else np.nan
            ax.bar(k + off, v, 0.36, color=col_)
    ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in steps], fontsize=8); ax.set_ylim(0, 100); ax.set_ylabel("agreement with reference (%)")
    ax.set_title("c  Symmetry ladder (symprec 0.1)", loc="left", fontsize=9)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#0072B2", label=f"matched ({int(matched.sum())})"), plt.Rectangle((0, 0), 1, 1, color="#D55E00", label=f"unmatched ({int((~matched).sum())})")], frameon=False, fontsize=7.6, loc="upper right")
    # d: dE
    ax = axs[3]; de = gd.dE_free.dropna().clip(-0.5, 1.0)
    if len(de): ax.hist(de, bins=np.linspace(-0.5, 1.0, 31), color=color, alpha=0.9)
    ax.axvline(-0.005, color=INK, lw=1, ls=(0, (3, 2))); ax.set_xlabel("dE: relaxed raw - relaxed ref (eV/atom)"); ax.set_ylabel("targets")
    ax.set_title(f"d  Energy after free relax (n={len(de)})", loc="left", fontsize=9)
    # e-h: strain / chemistry distributions, matched vs unmatched
    def hist(ax, col_, lo, hi, xl, ttl, absval=False):
        for sel, c, l in ((matched, "#0072B2", "matched"), (~matched, "#D55E00", "unmatched")):
            v = gd[sel][col_].astype(float).dropna(); v = v.abs() if absval else v
            if len(v): ax.hist(v.clip(lo, hi), bins=np.linspace(lo, hi, 26), color=c, alpha=0.65, label=l)
        ax.set_xlabel(xl); ax.set_ylabel("targets"); ax.set_title(ttl, loc="left", fontsize=9)
    hist(axs[4], "eps_vol", -1.0, 1.0, "eps_vol = ln det F (volume bias)", "e  Volume strain")
    hist(axs[5], "eps_shape", 0, 1.0, "eps_shape (deviatoric Hencky norm)", "f  Shape strain")
    hist(axs[6], "delta_cdm", -0.4, 0.4, "delta CDM (pred - reference)", "g  Cubic-deviation bias")
    hist(axs[7], "cn_accuracy", 0, 1.0, "coordination-number accuracy", "h  Local chemistry")
    for a in axs[4:]: a.legend(frameon=False, fontsize=7.4)
    for a in axs: a.grid(color=GRID, lw=0.6, alpha=0.7); a.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); return fig


def main():
    style(); df, Z, order, topk, summ, raw, fam = load()
    print("leaf order:", order)
    cl = fcluster(Z, 0.5, "distance"); print("cut 0.5:", {c: [a for a, x in zip(df.algorithm, cl) if x == c] for c in sorted(set(cl))})
    fig_dendrogram(df, Z); fig_overall(order, topk, summ, fam); fig_buckets(order, raw, summ, fam)
    fig_cluster_profile(order, raw, summ, fam); fig_heatmap(order, raw, summ, topk, fam)
    with PdfPages(OUT / "supp_algorithm_cards.pdf") as pdf:
        for name in order:
            fg = card(name, fam, raw, topk, summ); fg.savefig(OUT / "cards" / f"card_{KEY[name]}.png", dpi=170, facecolor="white"); pdf.savefig(fg, facecolor="white"); plt.close(fg)
    print("done")


if __name__ == "__main__":
    main()
