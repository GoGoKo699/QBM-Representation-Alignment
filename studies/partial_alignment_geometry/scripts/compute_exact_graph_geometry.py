from pathlib import Path
import json,sys
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs';sys.path.insert(0,str(REPOSITORY/'src'));sys.path.insert(0,str(STUDY/'scripts'))
from qbm_alignment.certificate_family import generate_family,evaluate,eff_condition
from partial_alignment_study import build_problem,canonical_biased_theta,SEEDS
instances=generate_family(INSTANCES);graphs=json.loads((GRAPHS/'partial_graphs.json').read_text());rows=[]
for inst in instances:
 for graph in ('chain','problem_tree','width2','width3','full'):
  P=build_problem(inst,graph,graphs)
  for state,seed,theta in [('projected_target',-1,P.c),*[(f'biased_seed_{s}',s,canonical_biased_theta(P,s)) for s in SEEDS]]:
   E,g,p,I=evaluate(theta,P.F,P.C,True);cond,rank,mn,mx=eff_condition(I);d=-np.linalg.pinv(I,rcond=1e-10)@g
   rows.append({'instance_id':inst.instance_id,'instance_width':inst.width,'graph':graph,'state':state,'seed':seed,'dimension':P.F.shape[1],'energy_gap':E-P.ground,'pstar':p[P.pidx],'gradient_rms':np.sqrt(np.mean(g*g)),'fisher_condition':cond,'fisher_rank':rank,'natural_target_cosine':float(d@P.c/(np.linalg.norm(d)*np.linalg.norm(P.c)))})
 print('done',inst.instance_id,flush=True)
pd.DataFrame(rows).to_csv(RESULTS/'exact_graph_geometry.csv',index=False)
