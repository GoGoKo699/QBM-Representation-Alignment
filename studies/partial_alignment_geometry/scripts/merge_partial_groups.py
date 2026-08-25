from pathlib import Path
import json,time
import pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs';G=RESULTS/'groups'
traj_files=sorted(G.glob('*_trajectories.csv'));log_files=sorted(G.glob('*_logs.csv.gz'))
if len(traj_files)!=120 or len(log_files)!=120:raise RuntimeError(f'expected 120 groups, found {len(traj_files)} / {len(log_files)}')
traj=pd.concat([pd.read_csv(f) for f in traj_files],ignore_index=True)
logs=pd.concat([pd.read_csv(f) for f in log_files],ignore_index=True)
traj.to_csv(RESULTS/'partial_alignment_trajectories.csv',index=False)
logs.to_csv(RESULTS/'partial_alignment_logs.csv.gz',index=False,compression='gzip')
meta={'groups':len(traj_files),'trajectories':len(traj),'logs':len(logs),'main_budget':256,'scaling_budgets':[64,1024],'steps':200,'parameter_seeds':[0,19,42,50,101],'graphs':['chain','problem_tree','width2','width3'],'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(RESULTS/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(meta)
