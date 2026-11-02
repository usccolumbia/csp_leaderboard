"""Main T180 table, pairwise decomposition against TCSPv3, prototype availability of generator-only successes and top-1 by arity."""
import json, pandas as pd, numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent; R=str(HERE/'data')+'/'; O=HERE/'output'/'tables'; O.mkdir(parents=True,exist_ok=True); O=str(O)+'/'
d=pd.read_csv(R+'t180_topk_sg_per_case_all.csv')
lib=json.load(open(R+'library_prototype_counts.json')); mp=json.load(open(R+'mp20_prototype_counts.json'))
NAMES={'tcspv3':'TCSPv3','crystalflow':'CrystalFlow','crystalite':'Crystalite','omatg':'OMatG','uni3dar':'Uni-3DAR','uflowcsp':'uFlowCSP','equicsp':'EquiCSP','mattergen':'MatterGen','diffcsp':'DiffCSP','cond-cdvae':'cond-CDVAE','gemsdiff':'GemsDiff','tgdmat':'TGDMat','symmcd':'SymmCD','crystalgrw':'CrystalGRW','crystalformer':'CrystalFormer','cdvae':'CDVAE'}
d=d[d.algorithm.isin(NAMES)]
d['sm1']=d.sm_rank.le(1); d['sm5']=d.sm_rank.le(5); d['sm20']=d.sm_rank.le(20)
d['sg1']=d.top1_sg1.fillna(False).astype(bool); d['sg01']=d.top1_sg01.fillna(False).astype(bool)
d['cons1']=d.sm1&d.sg1
rows=[]
for a,g in d.groupby('algorithm'):
    n=180
    rows.append(dict(algorithm=NAMES[a],coverage=int((g.n_cifs>0).sum()),top1=int(g.sm1.sum()),top5=int(g.sm5.sum()),top20=int(g.sm20.sum()),sg_top1=int(g.sg1.sum()),consensus_top1=int(g.cons1.sum()),sg_top1_symprec001=int(g.sg01.sum()),
        top1_pct=round(100*g.sm1.sum()/n,1),top20_pct=round(100*g.sm20.sum()/n,1),sg_pct=round(100*g.sg1.sum()/n,1),cons_pct=round(100*g.cons1.sum()/n,1)))
t=pd.DataFrame(rows).sort_values('top1',ascending=False); t.to_csv(O+'t180_main_table.csv',index=False); print(t.to_string(index=False))
P=d.pivot(index='material_id',columns='algorithm')
def dec(metric,k=None):
    base=P[metric]['tcspv3']; out=[]
    for a in NAMES:
        if a=='tcspv3': continue
        x=P[metric][a]
        out.append(dict(algorithm=NAMES[a],both=int((base&x).sum()),tcsp_only=int((base&~x).sum()),algo_only=int((~base&x).sum()),neither=int((~base&~x).sum())))
    df=pd.DataFrame(out); df.to_csv(O+f't180_decomp_{metric}.csv',index=False); return df
for m in ['sm1','sg1','cons1','sm20']:
    print(m); print(dec(m).to_string(index=False))
# union algo-only
for m in ['sm1','cons1','sm20']:
    base=P[m]['tcspv3']; u=np.zeros(len(base),bool)
    for a in NAMES:
        if a!='tcspv3': u|=P[m][a].values
    print('union algo-only',m,int((~base.values&u).sum()),'of 180; TCSPv3',int(base.sum()),'union of generators',int(u.sum()),' union incl TCSPv3',int((u|base.values).sum()))
# prototype availability for algo-only top-1 SM
base=P['sm1']['tcspv3']; rows=[]
for a in NAMES:
    if a=='tcspv3': continue
    ids=[m for m in P.index if (not base[m]) and P['sm1'][a][m]]
    c=dict(algorithm=NAMES[a],algo_only=len(ids),mp20=0,library_only=0,none=0)
    for m in ids:
        if mp[m]['n_same_sg']>0: c['mp20']+=1
        elif lib[m]['same_sg_excl_self']>0: c['library_only']+=1
        else: c['none']+=1
    rows.append(c)
pt=pd.DataFrame(rows).sort_values('algo_only',ascending=False); tot=pt.sum(numeric_only=True); print(pt.to_string(index=False)); print(tot.to_dict())
pt.to_csv(O+'t180_prototype_availability.csv',index=False)
# which targets have none, per algo
for a in NAMES:
    if a=='tcspv3': continue
    ids=[m for m in P.index if (not base[m]) and P['sm1'][a][m] and mp[m]['n_same_sg']==0 and lib[m]['same_sg_excl_self']==0]
    if ids: print('none:',a,ids)
# targets failing TCSPv3 top1 / top20
print('TCSPv3 top1 fail',int((~base).sum()),' top20 fail',int((~P['sm20']['tcspv3']).sum()))
# arity breakdown
md=pd.read_csv(HERE.parent/'testsets'/'T180'/'metadata.csv').set_index('material_id')
ar=md.nelements.reindex(P.index)
for a in NAMES:
    print(NAMES[a],[int(P['sm1'][a][ar==k].sum()) for k in (2,3,4)])
