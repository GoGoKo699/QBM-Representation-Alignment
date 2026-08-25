import sys
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / 'src'))
STUDY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / 'results' / 'boundary_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'
OUT = RESULTS
import multiprocessing as mp
import pickle
import pandas as pd
from qbm_alignment.certificate_family import generate_family
from qbm_alignment.optimizer_geometry import build, intervention

METHODS=('adam','armijo_gd','target_direction','ray_projected','projected_adam','exact_natural','damped_natural','diagonal_fisher')

def worker(payload):
    inst,checkpoint,theta,method=payload
    P=build(inst)
    budget=500 if method=='damped_natural' else 1000
    summary,logs=intervention(P,theta,method,budget,25)
    summary.update(replay_checkpoint=checkpoint,replay_budget=budget)
    for row in logs:row.update(replay_checkpoint=checkpoint,replay_budget=budget)
    return summary,logs

def main():
    family=generate_family(INSTANCES); im={x.instance_id:x for x in family}
    with (OUT/'baseline_states.pkl').open('rb') as f:states=pickle.load(f)
    payload=[(im[iid],cp,states[iid][cp],method) for iid in states for cp in (199,999) for method in METHODS]
    with mp.get_context('fork').Pool(8) as pool:outputs=list(pool.imap_unordered(worker,payload))
    summary=pd.DataFrame([x[0] for x in outputs]).sort_values(['instance_id','replay_checkpoint','optimizer'])
    logs=pd.DataFrame([r for x in outputs for r in x[1]]).sort_values(['instance_id','replay_checkpoint','optimizer','step'])
    summary.to_csv(OUT/'trap_replay_summary.csv',index=False); logs.to_csv(OUT/'trap_replay_geometry.csv',index=False)
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
