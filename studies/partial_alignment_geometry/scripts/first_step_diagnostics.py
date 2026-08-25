from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs'
sys.path.insert(0,str(STUDY/'scripts'));sys.path.insert(0,str(REPOSITORY/'src'))
from partial_alignment_study import (
    SEEDS, MAIN_METHODS, MAIN_BUDGET, Adam, build_problem, canonical_biased_theta,
    exact_natural_direction, sampled_update, exact_batch, stream_seed, generate_family,
    cosine,
)

def main():
    family=generate_family(INSTANCES)
    graphs=json.loads((GRAPHS/'partial_graphs.json').read_text())
    rows=[]
    for instance in family:
        for graph in ('chain','problem_tree','width2','width3'):
            problem=build_problem(instance,graph,graphs)
            for seed in SEEDS:
                theta=canonical_biased_theta(problem,seed)
                exact=exact_natural_direction(problem,theta)
                energy,probability,gradient=exact_batch(problem,theta[:,None])
                for method in MAIN_METHODS:
                    rng=np.random.default_rng(stream_seed('partial',problem.instance_id,problem.graph,seed,MAIN_BUDGET))
                    adam=Adam(len(theta)) if method=='sampled_adam' else None
                    direction,moments=sampled_update(problem,theta,method,probability[:,0],rng,MAIN_BUDGET,adam,True)
                    rows.append({
                        'instance_id':instance.instance_id,'instance_width':instance.width,
                        'split':'calibration' if instance.instance_id.endswith('_i1') else 'evaluation',
                        'graph':graph,'seed':seed,'sample_budget':MAIN_BUDGET,'method':method,
                        'parameter_dimension':len(theta),'sample_rank':moments.sample_rank,
                        'sample_alignment_residual':moments.alignment_residual,
                        'direction_cosine_exact_natural':cosine(direction,exact),
                        'direction_cosine_projected_target':cosine(direction,problem.c),
                        'exact_natural_cosine_projected_target':cosine(exact,problem.c),
                        'direction_relative_error_exact_natural':float(np.linalg.norm(direction-exact)/max(np.linalg.norm(exact),1e-300)),
                        'exact_natural_norm':float(np.linalg.norm(exact)),
                        'direction_norm':float(np.linalg.norm(direction)),
                        'initial_gap':float(energy[0]-problem.ground),
                    })
            print('done',instance.instance_id,graph,flush=True)
    pd.DataFrame(rows).to_csv(RESULTS/'first_step_diagnostics.csv',index=False)
if __name__=='__main__':main()
