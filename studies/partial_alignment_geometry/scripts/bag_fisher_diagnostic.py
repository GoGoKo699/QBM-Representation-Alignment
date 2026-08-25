from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs';sys.path.insert(0,str(REPOSITORY/'src'));sys.path.insert(0,str(STUDY/'scripts'))
from partial_alignment_study import SEEDS,build_problem,canonical_biased_theta,generate_family,regularized_solve,cap,cosine,SHRINK,STEP_CAP
from qbm_alignment.certificate_family import evaluate

def bags(problem,payload):
    order=tuple(payload[problem.instance_id][problem.graph]['order']);adj={i:set() for i in range(problem.n)}
    for a,b in problem.edges:adj[a].add(b);adj[b].add(a)
    rem=set(range(problem.n));out=[]
    for v in order:
        nb=sorted(adj[v]&rem);bag=set([v,*nb]);
        for i,a in enumerate(nb):
            for b in nb[i+1:]:adj[a].add(b);adj[b].add(a)
        rem.remove(v)
        supports=[{i} for i in range(problem.n)]+[set(e) for e in problem.edges]
        idx=tuple(i for i,s in enumerate(supports) if s<=bag and v in s)
        # assign each feature to earliest-eliminated endpoint bag; fields naturally assigned once.
        if idx:out.append(idx)
    # Nonoverlapping assignment gives block diagonal; also construct overlapping all-contained below separately.
    return tuple(out)

def overlap_bags(problem,payload):
    order=tuple(payload[problem.instance_id][problem.graph]['order']);adj={i:set() for i in range(problem.n)}
    for a,b in problem.edges:adj[a].add(b);adj[b].add(a)
    rem=set(range(problem.n));supports=[{i} for i in range(problem.n)]+[set(e) for e in problem.edges];out=[]
    for v in order:
        nb=sorted(adj[v]&rem);bag=set([v,*nb]);
        for i,a in enumerate(nb):
            for b in nb[i+1:]:adj[a].add(b);adj[b].add(a)
        rem.remove(v);idx=tuple(i for i,s in enumerate(supports) if s<=bag)
        if idx:out.append(idx)
    return tuple(out)

def direction(I,g,blocks,c):
    d=np.zeros_like(g);mult=np.zeros_like(g)
    for block in blocks:
        idx=np.asarray(block,int);d[idx]+=regularized_solve(I[np.ix_(idx,idx)],g[idx],SHRINK);mult[idx]+=1
    return cap(d/np.maximum(mult,1),c,STEP_CAP)

def main():
    fam=generate_family(INSTANCES);payload=json.loads((GRAPHS/'partial_graphs.json').read_text());rows=[]
    for inst in fam:
      for graph in ('chain','problem_tree','width2','width3'):
       p=build_problem(inst,graph,payload);blocks=overlap_bags(p,payload);storage=sum(len(b)*(len(b)+1)//2 for b in blocks);maxblock=max(map(len,blocks))
       for seed in SEEDS:
        th=canonical_biased_theta(p,seed);E,g,prob,I=evaluate(th,p.F,p.C,True);vals,vecs=np.linalg.eigh((I+I.T)/2);keep=vals>1e-10*max(float(vals[-1]),0);nat=-(vecs[:,keep]@((vecs[:,keep].T@g)/vals[keep]));d=direction(I,g,blocks,p.c)
        rows.append({'instance_id':p.instance_id,'instance_width':p.instance_width,'split':p.split,'graph':graph,'seed':seed,'blocks':len(blocks),'max_block_size':maxblock,'stored_entries':storage,'cosine_exact_natural':cosine(d,nat),'cosine_target':cosine(d,p.c),'relative_error':float(np.linalg.norm(d-nat)/max(np.linalg.norm(nat),1e-300))})
       print('done',p.instance_id,graph,flush=True)
    d=pd.DataFrame(rows);d.to_csv(RESULTS/'bag_fisher_diagnostic.csv',index=False);a=d[d.split=='evaluation'].groupby('graph').agg(n=('instance_id','size'),cos_mean=('cosine_exact_natural','mean'),cos_median=('cosine_exact_natural','median'),relerr_median=('relative_error','median'),storage_median=('stored_entries','median'),maxblock=('max_block_size','max')).reset_index();a.to_csv(RESULTS/'bag_fisher_diagnostic_summary.csv',index=False);print(a.to_string(index=False))
if __name__=='__main__':main()
