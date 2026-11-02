"""Numbers quoted in the paper text (T180; TCSPv3 relaxed and ranked with ORB-v3 like the generators): top-k of TCSPv3, paired tests
against every generator, complementarity at top-1 and top-20, and prototype availability of generator-only successes.
Run figure_failure_regimes.py first; its values are printed at the end."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent; M = HERE / "data"
GENS = ["crystalflow","crystalite","omatg","uni3dar","equicsp","uflowcsp","mattergen","diffcsp","cond-cdvae","gemsdiff","tgdmat","cdvae","symmcd","crystalgrw","crystalformer"]
mp20 = json.load(open(M / "mp20_prototype_counts.json")); lib = json.load(open(M / "library_prototype_counts.json"))
def run(label, topk_file):
    t = pd.read_csv(topk_file); ids = sorted(t[t.algorithm == "tcspv3"].material_id)
    rk = {a: pd.to_numeric(t[t.algorithm == a].set_index("material_id").sm_rank, errors="coerce").reindex(ids) for a in ["tcspv3"] + GENS}
    hit = lambda a, k: (rk[a] <= k).to_numpy()
    T1, T20 = hit("tcspv3", 1), hit("tcspv3", 20)
    print(f"== {label}: TCSPv3 top1/5/20", int(T1.sum()), int(hit("tcspv3", 5).sum()), int(T20.sum()))
    miss = ~T1; rec = miss & T20; print("  top-1 misses", int(miss.sum()), "recovered within top-20", int(rec.sum()), "median rank of recovered", float(np.median(rk["tcspv3"][rec])))
    p = {}
    for a in GENS:
        x, y = T1, hit(a, 1); b, c = int((x & ~y).sum()), int((~x & y).sum()); p[a] = (binomtest(b, b + c, .5).pvalue if b + c else 1.0, b, c)
    order = sorted(p, key=lambda a: p[a][0]); run_ = 0; holm = {}
    for i, a in enumerate(order): run_ = max(run_, min(1.0, p[a][0] * (len(order) - i))); holm[a] = run_
    for a in GENS[:8]: print(f"  top-1 vs {a:12s} only-T {p[a][1]:3d} only-G {p[a][2]:3d} p={p[a][0]:.3g} p_Holm={holm[a]:.3g}")
    print("  max p_Holm over all 15:", round(max(holm.values()), 4))
    p20 = {}
    for a in GENS:
        x, y = T20, hit(a, 20); b, c = int((x & ~y).sum()), int((~x & y).sum()); p20[a] = binomtest(b, b + c, .5).pvalue if b + c else 1
    print("  top-20 max raw p:", f"{max(p20.values()):.2g}")
    G1 = np.any([hit(a, 1) for a in GENS], axis=0); G20 = np.any([hit(a, 20) for a in GENS], axis=0)
    print("  top-1: gens-not-TCSP", int((G1 & ~T1).sum()), "TCSP-alone", int((T1 & ~G1).sum()), "| top-20: gens-not-TCSP", int((G20 & ~T20).sum()), "TCSP-alone", int((T20 & ~G20).sum()), "union gens top20", int(G20.sum()))
    succ = [(i, a) for a in GENS for j, i in enumerate(ids) if hit(a, 1)[j] and not T1[j]]
    cat = {"mp20": 0, "lib": 0, "none": 0}; none_t = set()
    for i, a in succ:
        if mp20[i]["n_same_sg"] > 0: cat["mp20"] += 1
        elif lib[i]["same_sg_excl_self"] > 0: cat["lib"] += 1
        else: cat["none"] += 1; none_t.add(i)
    print("  gen-only top-1 successes", len(succ), "targets", len({i for i, _ in succ}), cat, "none targets", none_t)
run("T180", M / "t180_topk_sg_per_case_all.csv")
raw = pd.read_csv(M / "raw_top1_cascade.csv"); g = raw[raw.algorithm == "tcspv3"]; mt = g[g.match_primary_raw == True]
print("TCSPv3 raw top-1 match", int((g.match_primary_raw == True).sum()), "of", len(g), "| matched raw with SG (0.1):", round(100 * mt.sg_ok.astype(bool).mean(), 1))
print("TCSPv3 raw buckets", g.bucket_primary.value_counts().to_dict())
v = json.load(open(HERE / "output/paper_figures/fig_failure_regimes_values.json"))
for k, val in v.items():
    if "TCSPv3" in k: print(k, val)
