import argparse,pickle,time
from pathlib import Path
import run_independent_full_fisher as r
ap=argparse.ArgumentParser();ap.add_argument('--start',type=int);ap.add_argument('--end',type=int);a=ap.parse_args()
fam=r.generate_family(r.INSTANCES);tasks=[(i,M) for i in fam for M in r.BUDGETS];st=time.time()
for k in range(a.start,min(a.end,len(tasks))):
 inst,M=tasks[k];p=r.TASK/f'{inst.instance_id}_M{M}.pkl'
 if not p.exists():p.write_bytes(pickle.dumps(r.worker((inst,M)),protocol=5))
 print(k+1,'/',len(tasks),'elapsed',time.time()-st,flush=True)
