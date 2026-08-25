from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs';sys.path.insert(0,str(REPOSITORY/'src'));sys.path.insert(0,str(STUDY/'scripts'))
from partial_alignment_study import *
METHODS=('sampled_diagonal_fisher','sampled_two_block_fisher','sampled_star_fisher','sampled_full_fisher','protected_ray_star')

def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-15))
def main():
 fam=generate_family(INSTANCES);gp=json.loads((GRAPHS/'partial_graphs.json').read_text());rows=[]
 for inst in fam:
  if inst.instance_id.endswith('_i1'):continue
  for graph in GRAPH_NAMES:
   P=build_problem(inst,graph,gp)
   for seed in SEEDS:
    theta=canonical_biased_theta(P,seed);E,g,p,I=evaluate(theta,P.F,P.C,True);exact=-np.linalg.pinv(I,rcond=1e-10)@g
    stream=stream_seed('partial',P.instance_id,P.graph,seed,256);rng=np.random.default_rng(stream);mom=sample_moments(P,p,rng,256,True,True)
    directions={}
    ridge=RIDGE*max(float(mom.diagonal.max()),1e-12);directions['sampled_diagonal_fisher']=cap(-mom.gradient/(mom.diagonal+ridge),P.c,STEP_CAP)
    directions['sampled_two_block_fisher']=cap(two_block_direction(P,mom,mom.gradient),P.c,STEP_CAP)
    directions['sampled_star_fisher']=cap(star_direction(P,mom,mom.gradient),P.c,STEP_CAP)
    directions['sampled_full_fisher']=cap(full_direction(mom,mom.gradient),P.c,STEP_CAP)
    # protected ray first update
    c2=float(P.c@P.c);beta=float(theta@P.c/c2);u=theta-beta*P.c;proj=mom.centered@P.c;Ib=float(proj@proj/255);gb=float(mom.gradient@P.c);scale=max(float(mom.diagonal.mean()),1e-12);db=float(np.clip(-gb/(Ib+RIDGE*scale*c2),-RAY_CAP,RAY_CAP));rg=mom.gradient-(gb/c2)*P.c;res=star_direction(P,mom,rg);res-=float(res@P.c/c2)*P.c;res=cap(res,P.c,RESIDUAL_CAP);directions['protected_ray_star']=(beta+db)*P.c+TRANSVERSE_KEEP*u+res-theta
    exact_align=float(np.linalg.norm(g+I@P.c)/max(np.linalg.norm(g),1e-15));var=float(p@((P.C-E)**2));expl=float(g@np.linalg.pinv(I,rcond=1e-10)@g/var)
    for method,d in directions.items():
     rows.append({'instance_id':P.instance_id,'instance_width':P.instance_width,'graph':graph,'seed':seed,'method':method,'parameter_dimension':len(P.c),'exact_ng_norm':float(np.linalg.norm(exact)),'exact_ng_target_cosine':cosine(exact,P.c),'exact_alignment_residual':exact_align,'exact_explained_variance':expl,'sample_rank':mom.sample_rank,'sample_alignment_residual':mom.alignment_residual,'direction_cosine_exact_ng':cosine(d,exact),'direction_relative_error_exact_ng':rel(d,cap(exact,P.c,STEP_CAP)),'direction_cosine_target':cosine(d,P.c)})
  print('done',inst.instance_id,flush=True)
 pd.DataFrame(rows).to_csv(RESULTS/'initial_direction_diagnostics.csv',index=False)
 print('wrote',len(rows))
if __name__=='__main__':main()
