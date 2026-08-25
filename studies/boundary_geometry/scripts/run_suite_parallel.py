import sys
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / 'src'))
STUDY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / 'results' / 'boundary_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'
OUT = RESULTS
import itertools,multiprocessing as mp
import pandas as pd
import numpy as np
from qbm_alignment.certificate_family import generate_family
from qbm_alignment.optimizer_geometry import build, intervention, SEEDS
METHODS=('adam','armijo_gd','target_direction','ray_projected','projected_adam','exact_natural','diagonal_fisher')

def worker(payload):
    inst,init,seed,method=payload;P=build(inst)
    
    if init=='exact_target':theta=P.c.copy()
    else:
        pairs=tuple(itertools.combinations(range(P.instance.n),2));index={edge:i for i,edge in enumerate(pairs)}
        full=.3*np.random.default_rng(seed).standard_normal(P.instance.n+len(pairs));active=np.asarray(list(range(P.instance.n))+[P.instance.n+index[e] for e in P.instance.edges],dtype=np.int64);theta=P.c+full[active]
    summary,logs=intervention(P,theta,method,200,50)
    summary.update(initialization=init,seed=seed)
    for row in logs:row.update(initialization=init,seed=seed)
    return summary,logs

def main():
    family=generate_family(INSTANCES);payload=[]
    for inst in family:
        for method in METHODS:payload.append((inst,'exact_target',-1,method))
        for seed in SEEDS:
            for method in METHODS:payload.append((inst,'target_biased',seed,method))
    with mp.get_context('fork').Pool(12) as pool:outputs=list(pool.imap_unordered(worker,payload,chunksize=1))
    summary=pd.DataFrame([x[0] for x in outputs]).sort_values(['initialization','instance_id','seed','optimizer'])
    logs=pd.DataFrame([r for x in outputs for r in x[1]]).sort_values(['initialization','instance_id','seed','optimizer','step'])
    summary.to_csv(OUT/'optimizer_suite_summary.csv',index=False);logs.to_csv(OUT/'optimizer_suite_geometry.csv',index=False)
    print(summary.groupby(['initialization','optimizer']).agg(trajectories=('success','size'),successes=('success','sum'),rate=('success','mean'),mean_first=('first_success',lambda x:x[x>=0].mean()),median_gap=('minimum_gap','median'),evals=('gradient_evaluations','mean')).to_string(),flush=True)
if __name__=='__main__':main()
