"""Technique feature matrix + clustering figures for the CSP algorithms.

Outputs (output/technique_clustering/):
  feature_matrix.csv              algorithm x technique features, family, confidence, notes
  technique_dendrogram.{pdf,png}  hierarchical clustering of the feature matrix (Jaccard, average linkage)
  technique_clustermap.{pdf,png}  dendrogram + the feature matrix itself

Colors: fixed-order categorical hues from the dataviz reference palette (validated, light mode)
with a marker shape per family as secondary encoding; neutral ink for filled cells; a single
blue ramp for the overlap magnitudes.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

OUT = Path(__file__).resolve().parent / "output" / "technique_clustering"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
FAMILIES = [  # (label, hex, marker): fixed palette order, never cycled
    ("Template substitution", "#2a78d6", "o"),
    ("Evolutionary + MLIP", "#eb6834", "s"),
    ("Diffusion / score-based", "#1baf7a", "^"),
    ("Flow / ODE", "#eda100", "D"),
    ("Symmetry-first (Wyckoff)", "#e87ba4", "P"),
    ("Token sequence models", "#008300", "X"),
    ("VAE + Langevin", "#4a3aa7", "v"),
]
FEATURE_GROUPS = [
    ("Paradigm", ["Template substitution", "Evolutionary search", "Learned generative"]),
    ("Mechanism", ["Diffusion / score", "Flow / ODE", "VAE + Langevin", "Autoregressive", "Masked-token"]),
    ("Representation", ["Continuous lattice + coords", "Wyckoff / asym. unit", "3D octree tokens"]),
    ("Symmetry & cond.", ["Explicit space group", "Text-conditioned"]),
]
FEATURES = [f for _, fs in FEATURE_GROUPS for f in fs]

# name, family index, confidence, features on, basis for the assignment
ALGOS = [
    ("TCSP 2.0", 0, "high", ["Template substitution"], "TCSP2.0 methods notes: template library + BERTOS + embedding similarity"),
    ("TCSPv3", 0, "high", ["Template substitution"], "TCSPV3 README/ALGORITHM.md: retrieval + multi-assignment substitution"),
    ("TCSP3p2", 0, "high", ["Template substitution"], "TCSP3p2 README: cross-stoichiometry framework decoration"),
    ("CSPML", 0, "high", ["Template substitution"], "dispatcher: learned template similarity + substitution"),
    ("MAGUS", 1, "high", ["Evolutionary search", "Continuous lattice + coords", "Explicit space group"], "genetic algorithm, spacegroup-constrained generation"),
    ("USPEX", 1, "high", ["Evolutionary search", "Continuous lattice + coords", "Explicit space group"], "genetic algorithm, spacegroup-constrained generation"),
    ("CDVAE", 6, "high", ["Learned generative", "Diffusion / score", "VAE + Langevin", "Continuous lattice + coords"], "VAE with score-based Langevin decoding"),
    ("cond-CDVAE", 6, "medium", ["Learned generative", "Diffusion / score", "VAE + Langevin", "Continuous lattice + coords"], "dispatcher: conditional CDVAE sampler"),
    ("DiffCSP", 2, "high", ["Learned generative", "Diffusion / score", "Continuous lattice + coords"], "joint diffusion over lattice and fractional coordinates"),
    ("EquiCSP", 2, "high", ["Learned generative", "Diffusion / score", "Continuous lattice + coords"], "equivariant diffusion CSP"),
    ("MatterGen", 2, "high", ["Learned generative", "Diffusion / score", "Continuous lattice + coords"], "diffusion over types, coordinates, lattice"),
    ("TGDMat", 2, "medium", ["Learned generative", "Diffusion / score", "Continuous lattice + coords", "Text-conditioned"], "dispatcher: text-conditioned joint diffusion"),
    ("Crystalite", 2, "high", ["Learned generative", "Diffusion / score", "Continuous lattice + coords"], "arXiv:2604.02270: lightweight diffusion transformer"),
    ("CrystalGRW", 2, "low", ["Learned generative", "Diffusion / score", "Continuous lattice + coords"], "dispatcher: graph random-walk diffusion sampler (mechanism not verified)"),
    ("FlowMM", 3, "high", ["Learned generative", "Flow / ODE", "Continuous lattice + coords"], "dispatcher: Riemannian flow matching"),
    ("CrystalFlow", 3, "high", ["Learned generative", "Flow / ODE", "Continuous lattice + coords"], "flow-matching CSP"),
    ("OMatG", 3, "medium", ["Learned generative", "Flow / ODE", "Continuous lattice + coords"], "dispatcher: formula-conditioned ODE generator"),
    ("uFlowCSP", 3, "high", ["Learned generative", "Flow / ODE", "Continuous lattice + coords"], "MeanFlow few-step flow (our checkpoint)"),
    ("DiffCSP++", 4, "high", ["Learned generative", "Diffusion / score", "Continuous lattice + coords", "Wyckoff / asym. unit", "Explicit space group"], "dispatcher: space-group / Wyckoff-conditioned diffusion"),
    ("SymmCD", 4, "medium", ["Learned generative", "Diffusion / score", "Continuous lattice + coords", "Wyckoff / asym. unit", "Explicit space group"], "dispatcher: symmetry-aware conditional diffusion"),
    ("GemsDiff", 4, "medium", ["Learned generative", "Diffusion / score", "Continuous lattice + coords", "Wyckoff / asym. unit", "Explicit space group"], "web search: space-group classifier, lattice diffusion, asymmetric-unit diffusion"),
    ("CrystalFormer", 5, "high", ["Learned generative", "Autoregressive", "Wyckoff / asym. unit", "Explicit space group"], "autoregressive transformer over space-group / Wyckoff tokens"),
    ("Uni-3DAR", 5, "medium", ["Learned generative", "Autoregressive", "3D octree tokens"], "dispatcher: autoregressive CSP; octree tokenization from the method's paper"),
    ("MaskGXT", 5, "medium", ["Learned generative", "Masked-token", "Wyckoff / asym. unit", "Explicit space group"], "arXiv:2606.22866 abstract: masked fill-in over space group, Wyckoff, lattice and coordinate bins"),
]


def _style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300, "figure.dpi": 150})


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _family_legend(ax, loc, used):
    handles = [Line2D([], [], marker=m, linestyle="", markerfacecolor=c, markeredgecolor="white", markersize=8, label=lab)
               for i, (lab, c, m) in enumerate(FAMILIES) if i in used]
    return ax.legend(handles=handles, loc=loc, frameon=False, fontsize=8, title="Primary technique", title_fontsize=8,
                     handletextpad=0.4, labelspacing=0.5, borderaxespad=0.0)


def build_matrix() -> pd.DataFrame:
    rows = []
    for name, fam, conf, on, note in ALGOS:
        r = {"algorithm": name, "family": FAMILIES[fam][0], "family_index": fam, "confidence": conf}
        r.update({f: int(f in on) for f in FEATURES})
        r["basis"] = note
        rows.append(r)
    return pd.DataFrame(rows)


def cluster(df: pd.DataFrame):
    X = df[FEATURES].to_numpy(dtype=bool)
    D = pdist(X, "jaccard")
    return linkage(D, "average"), squareform(D)


def _label(row):
    return row.algorithm + ("*" if row.confidence == "low" else "")


def draw_dendrogram_axes(ax, Z, df, order_labels=True, cut=0.5):
    d = dendrogram(Z, no_plot=True)
    for ic, dc in zip(d["icoord"], d["dcoord"]):
        ax.plot(dc, ic, color=INK2, lw=1.0, solid_capstyle="round")
    leaves = d["leaves"]
    ypos = {leaf: 5 + 10 * k for k, leaf in enumerate(leaves)}
    ax.set_ylim(len(leaves) * 10, 0)
    ax.set_xlim(1.02, -0.02)
    ax.axvline(cut, color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax.set_xlabel("Jaccard distance between technique profiles")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return leaves, ypos


def fig_dendrogram(df, Z):
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    leaves, ypos = draw_dendrogram_axes(ax, Z, df)
    for leaf in leaves:
        r = df.iloc[leaf]
        _, color, marker = FAMILIES[r.family_index]
        y = ypos[leaf]
        ax.plot(-0.045, y, marker=marker, color=color, markersize=7.5, markeredgecolor="white", clip_on=False)
        ax.text(-0.085, y, _label(r), va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlim(1.02, -0.55)
    _family_legend(ax, "lower left", set(df.family_index))
    ax.text(0.5, 1.005, "cut at 0.5", transform=ax.get_xaxis_transform(), color=INK2, fontsize=8, ha="center", va="bottom")
    ax.annotate("* mechanism not independently verified", xy=(1.0, 0.0), xycoords="axes fraction", xytext=(0, -40),
                textcoords="offset points", fontsize=7.5, color=INK2, ha="right", va="top")
    _save(fig, "technique_dendrogram")
    return leaves


def fig_clustermap(df, Z):
    d = dendrogram(Z, no_plot=True)
    order = d["leaves"]
    n, m = len(order), len(FEATURES)
    fig = plt.figure(figsize=(8.6, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.06 + m * 0.34, 1.6], wspace=0.03, left=0.02, right=0.99, top=0.80, bottom=0.17)
    axd, axh, axl = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    draw_dendrogram_axes(axd, Z, df)
    axd.set_xlabel("Jaccard distance")
    M = df.iloc[order][FEATURES].to_numpy()
    from matplotlib.colors import to_rgb
    pal = {"Template substitution": "#D55E00", "Evolutionary search": "#E69F00", "Learned generative": "#0072B2",
           "Diffusion / score": "#009E73", "Flow / ODE": "#56B4E9", "VAE + Langevin": "#CC79A7", "Autoregressive": "#7B3294", "Masked-token": "#A6761D",
           "Continuous lattice + coords": "#4C78A8", "Wyckoff / asym. unit": "#F58518", "3D octree tokens": "#54A24B",
           "Explicit space group": "#B279A2", "Text-conditioned": "#E45756"}
    RGB = np.ones((n, m, 3))
    for j, f in enumerate(FEATURES):
        c = np.array(to_rgb(pal.get(f, "#3d3c39")))
        RGB[M[:, j] > 0.5, j, :] = c
    axh.imshow(RGB, aspect="auto", interpolation="nearest")
    axh.autoscale(False)
    axh.set_xticks(np.arange(m) - 0.5, minor=True); axh.set_yticks(np.arange(n) - 0.5, minor=True)
    axh.grid(which="minor", color="white", lw=1.4); axh.tick_params(which="both", length=0)
    axh.set_xticks(range(m)); axh.set_xticklabels(FEATURES, rotation=55, ha="left", fontsize=8, rotation_mode="anchor")
    axh.xaxis.tick_top(); axh.set_yticks([])
    for s in axh.spines.values():
        s.set_visible(False)
    col = 0
    for gname, fs in FEATURE_GROUPS:
        axh.plot([col - 0.45, col + len(fs) - 0.55], [n + 0.0, n + 0.0], color=INK2, lw=1.0, clip_on=False, transform=axh.transData)
        axh.text(col + (len(fs) - 1) / 2, n + 0.35, gname.replace(" & ", "\n& "), ha="center", va="top", fontsize=8.5, color=INK, fontweight="bold", clip_on=False)
        col += len(fs)
        if col < m:
            axh.axvline(col - 0.5, color=GRID, lw=1.2)
    axh.set_xlim(-0.5, m - 0.5); axh.set_ylim(n - 0.5, -0.5)
    axl.set_xlim(0, 1); axl.set_ylim(n - 0.5, -0.5); axl.axis("off")
    for k, leaf in enumerate(order):
        r = df.iloc[leaf]
        _, color, marker = FAMILIES[r.family_index]
        axl.plot(0.05, k, marker=marker, color=color, markersize=7, markeredgecolor="white", clip_on=False)
        axl.text(0.13, k, _label(r), va="center", ha="left", fontsize=9)
    fig.legend(handles=[Line2D([], [], marker=m_, linestyle="", markerfacecolor=c, markeredgecolor="white", markersize=8, label=lab)
                        for i, (lab, c, m_) in enumerate(FAMILIES) if i in set(df.family_index)],
               loc="upper center", bbox_to_anchor=(0.5, 0.075), ncol=4, frameon=False, fontsize=8, title="Primary technique", title_fontsize=8)
    _save(fig, "technique_clustermap")


def main():
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_matrix()
    df.to_csv(OUT / "feature_matrix.csv", index=False)
    Z, Dm = cluster(df)
    leaves = fig_dendrogram(df, Z)
    fig_clustermap(df, Z)
    print("feature-based leaf order:", [df.algorithm[i] for i in leaves])
    for t in (0.3, 0.5, 0.7):
        cl = fcluster(Z, t, "distance")
        print(f"cut {t}: {len(set(cl))} clusters ->", {c: [a for a, x in zip(df.algorithm, cl) if x == c] for c in sorted(set(cl))})


if __name__ == "__main__":
    main()
