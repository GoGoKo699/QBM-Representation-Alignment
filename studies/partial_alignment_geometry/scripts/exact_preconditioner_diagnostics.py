from __future__ import annotations
import json,sys,math
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs';sys.path.insert(0,str(REPOSITORY/'src'));sys.path.insert(0,str(STUDY/'scripts'))
from partial_alignment_study import SEEDS,build_problem,canonical_biased_theta,generate_family,regularized_solve,cap,cosine,SHRINK,FULL_SHRINK,STEP_CAP
from qbm_alignment.certificate_family import evaluate

def block_direction(problem,I,g):
    out=np.zeros_like(g)
    for sel in (np.arange(problem.n),np.arange(problem.n,len(g))):
        out[sel]=regularized_solve(I[np.ix_(sel,sel)],g[sel],SHRINK)
    return cap(out,problem.c,STEP_CAP)

def star_direction(problem,I,g):
    out=np.zeros_like(g)
    for block in problem.star_blocks:
        idx=np.asarray(block,int);out[idx]+=regularized_solve(I[np.ix_(idx,idx)],g[idx],SHRINK)
    return cap(out/problem.multiplicity,problem.c,STEP_CAP)

def main():
    fam=generate_family(INSTANCES);graphs=json.loads((GRAPHS/'partial_graphs.json').read_text());rows=[]
    for inst in fam:
      for graph in ('chain','problem_tree','width2','width3'):
       p=build_problem(inst,graph,graphs)
       for seed in SEEDS:
        th=canonical_biased_theta(p,seed);E,g,prob,I=evaluate(th,p.F,p.C,True);vals,vecs=np.linalg.eigh((I+I.T)/2);mx=max(float(vals[-1]),0);keep=vals>1e-10*mx;nat=-(vecs[:,keep]@((vecs[:,keep].T@g)/vals[keep]))
        dirs={
          'exact_natural':nat,
          'exact_regularized_full':cap(regularized_solve(I,g,FULL_SHRINK),p.c,STEP_CAP),
          'exact_diagonal':cap(-g/(np.diag(I)+1e-3*max(float(np.diag(I).max()),1e-12)),p.c,STEP_CAP),
          'exact_two_block':block_direction(p,I,g),
          'exact_star':star_direction(p,I,g),
          'projected_target':p.c,
        }
        for method,d in dirs.items():
          rows.append({'instance_id':p.instance_id,'instance_width':p.instance_width,'split':p.split,'graph':graph,'seed':seed,'method':method,'parameter_dimension':len(g),'gap':float(E-p.ground),'direction_cosine_exact_natural':cosine(d,nat),'direction_cosine_projected_target':cosine(d,p.c),'direction_relative_error_exact_natural':float(np.linalg.norm(d-nat)/max(np.linalg.norm(nat),1e-300)),'direction_norm':float(np.linalg.norm(d)),'exact_natural_norm':float(np.linalg.norm(nat)),'fisher_rank':int(keep.sum()),'fisher_condition':float(vals[-1]/vals[keep][0])})
       print('done',p.instance_id,graph,flush=True)
    d=pd.DataFrame(rows);d.to_csv(RESULTS/'exact_preconditioner_diagnostics.csv',index=False)
    e=d[d.split=='evaluation'];a=e.groupby(['graph','method'],as_index=False).agg(n=('instance_id','size'),cos_exact_mean=('direction_cosine_exact_natural','mean'),cos_exact_median=('direction_cosine_exact_natural','median'),relative_error_median=('direction_relative_error_exact_natural','median'),cos_target_mean=('direction_cosine_projected_target','mean'))
    a.to_csv(RESULTS/'exact_preconditioner_diagnostics_summary.csv',index=False);print(a.to_string(index=False))
if __name__=='__main__':main()
