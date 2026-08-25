from __future__ import annotations
import json,pickle,time
from pathlib import Path
import pandas as pd
import finite_sample_study as fs

TASKDIR=fs.OUT/'tasks';TASKDIR.mkdir(parents=True,exist_ok=True)

def broad_tasks():
    fam=fs.generate_family(fs.INSTANCES)
    return [(inst,budget) for inst in fam for budget in (0,)+fs.BUDGETS]

def trap_tasks():
    return [(iid,cp,budget) for iid in ('ct_w5_i1','ct_w6_i1') for cp in (199,999) for budget in fs.BUDGETS]

def broad_name(task):
    inst,b=task;return TASKDIR/f'broad_{inst.instance_id}_M{b}.pkl'

def trap_name(task):
    iid,cp,b=task;return TASKDIR/f'trap_{iid}_cp{cp}_M{b}.pkl'

def run_tasks(kind,start_index=0,end_index=None):
    tasks=broad_tasks() if kind=='broad' else trap_tasks()
    if end_index is None:end_index=len(tasks)
    start=time.time()
    done=0
    for index in range(start_index,min(end_index,len(tasks))):
        task=tasks[index];path=broad_name(task) if kind=='broad' else trap_name(task)
        if path.exists():
            done+=1;continue
        output=fs.broad_worker(task) if kind=='broad' else fs.trap_worker(task)
        path.write_bytes(pickle.dumps(output,protocol=5))
        done+=1
        print(kind,index+1,'/',len(tasks),'chunk_done',done,'elapsed',time.time()-start,flush=True)

def merge():
    brows=[];blogs=[]
    for task in broad_tasks():
        p=broad_name(task)
        if not p.exists():raise FileNotFoundError(p)
        rows,logs=pickle.loads(p.read_bytes());brows.extend(rows);blogs.extend(logs)
    pd.DataFrame(brows).to_csv(fs.OUT/'finite_sample_broad_trajectories.csv',index=False)
    pd.DataFrame(blogs).to_csv(fs.OUT/'finite_sample_broad_logs.csv.gz',index=False,compression='gzip')
    trows=[];tlogs=[]
    for task in trap_tasks():
        p=trap_name(task)
        if not p.exists():raise FileNotFoundError(p)
        rows,logs=pickle.loads(p.read_bytes());trows.extend(rows);tlogs.extend(logs)
    pd.DataFrame(trows).to_csv(fs.OUT/'finite_sample_trap_replays.csv',index=False)
    pd.DataFrame(tlogs).to_csv(fs.OUT/'finite_sample_trap_logs.csv.gz',index=False,compression='gzip')
    meta={'budgets':fs.BUDGETS,'parameter_seeds':fs.SEEDS,'methods':fs.METHODS+('analytic_target_cooling',),'steps':fs.STEPS,'calibration_instances':[f'ct_w{w}_i1' for w in (3,4,5,6)],'evaluation_instances':[f'ct_w{w}_i{i}' for w in (3,4,5,6) for i in range(2,6)],'trap_replicates':5,'broad_rows':len(brows),'trap_rows':len(trows)}
    (fs.OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(meta)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['broad','trap','merge']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=None);a=ap.parse_args()
    if a.mode=='merge':merge()
    else:run_tasks(a.mode,a.start,a.end)
