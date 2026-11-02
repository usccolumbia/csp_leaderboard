"""LaTeX bodies of the supplementary T180 tables for all 16 methods (coverage/strict, space-group tolerance, arity, algorithm-only targets)."""
import json, re
from pathlib import Path
import pandas as pd
from pymatgen.core import Composition
HERE = Path(__file__).resolve().parent; R = str(HERE / 'data') + '/'; OUT = HERE / 'output' / 'tables'; OUT.mkdir(parents=True, exist_ok=True)
NAMES = {'tcspv3':'TCSPv3','crystalflow':'CrystalFlow','crystalite':'Crystalite','omatg':'OMatG','uni3dar':'Uni-3DAR','equicsp':'EquiCSP','uflowcsp':'uFlowCSP','diffcsp':'DiffCSP','mattergen':'MatterGen','cond-cdvae':'cond-CDVAE','gemsdiff':'GemsDiff','tgdmat':'TGDMat','cdvae':'CDVAE','symmcd':'SymmCD','crystalgrw':'CrystalGRW','crystalformer':'CrystalFormer'}
d = pd.read_csv(R+'t180_topk_sg_per_case_all.csv'); d = d[d.algorithm.isin(NAMES)]
s = pd.read_csv(R+'t180_strict_primitive_per_case.csv')
q = json.load(open(HERE.parent / 'testsets/T180/queries.json'))
nel = {x['material_id']: len(Composition(x['formula']).elements) for x in q}
form = {x['material_id']: x['formula'] for x in q}
sgref = {r.material_id: r.gt_sg01 for r in d[d.algorithm == 'tcspv3'].itertuples()}
mp = json.load(open(R+'mp20_prototype_counts.json')); lib = json.load(open(R+'library_prototype_counts.json'))
def tex(f):
    c = Composition(f).formula.split(); out = []
    for t in c:
        m = re.match(r'([A-Z][a-z]?)(\d*\.?\d*)$', t); el, n = m.group(1), m.group(2)
        out.append(el + ('$_{%s}$' % n if n not in ('', '1') else ''))
    return ''.join(out)
def top(g, k): return int((g.sm_rank.notna() & (g.sm_rank <= k)).sum())
cov = []; sgt = []; ar = []
for a, nm in NAMES.items():
    g = d[d.algorithm == a]; gs = s[s.algorithm == a]
    full = int((g.n_cifs >= 20).sum()); none = int((g.n_cifs == 0).sum()); valid = 180 - none; part = valid - full
    st = [int((gs.strict_rank.notna() & (gs.strict_rank <= k)).sum()) for k in (1, 5, 20)]
    cov.append(f"{nm} & {valid} & {full} & {part} & {none} & {st[0]} & {st[1]} & {st[2]} \\\\")
    a1, a01 = int(g.top1_sg1.fillna(False).astype(bool).sum()), int(g.top1_sg01.fillna(False).astype(bool).sum())
    sgt.append(f"{nm} & {a1} ({100*a1/180:.1f}\\%) & {a01} ({100*a01/180:.1f}\\%) \\\\")
    h = g[g.sm_rank.eq(1)].material_id; cnt = [int(sum(nel[m] == k for m in h)) for k in (2, 3, 4)]
    ar.append(f"{nm} & {cnt[0]} & {cnt[1]} & {cnt[2]} \\\\")
P = d.pivot(index='material_id', columns='algorithm')
base = P['sm_rank']['tcspv3'].le(1)
rows = []
for m in P.index:
    solvers = [NAMES[a] for a in NAMES if a != 'tcspv3' and P['sm_rank'][a][m] <= 1]
    if solvers and not base[m]: rows.append((m, solvers))
rows.sort(key=lambda r: (-len(r[1]), r[0]))
alg = [f"{m} & {tex(form[m])} & {int(sgref[m])} & {', '.join(sv)} & {mp[m]['n_same_sg']} & {lib[m]['same_sg_excl_self']} \\\\" for m, sv in rows]
print(len(rows), sum(len(sv) for _, sv in rows))
json.dump(dict(cov=cov, sg=sgt, arity=ar, algo=alg), open(OUT / 'supp_tables.json', 'w'), indent=1)
