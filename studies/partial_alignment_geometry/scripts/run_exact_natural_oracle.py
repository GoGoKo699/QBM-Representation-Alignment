from __future__ import annotations
import argparse,json,math,multiprocessing as mp,sys,time
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs'
sys.path.insert(0,str(STUDY/'scripts'));sys.path.insert(0,str(REPOSITORY/'src'))
from partial_alignment_study import SEEDS,build_problem,canonical_biased_theta,generate_family,cosine
from qbm_alignment.certificate_family import evaluate,eff_condition,local_min

STEPS=200;TOL=.1;STEP_CAP=.5;ARMIJO=.0001;STAGNATION_WINDOW=15;STAGNATION_TOL=1e-10

def natural_direction(I,g):
    vals,vecs=np.linalg.eigh((I+I.T)/2);mx=max(float(vals[-1]),0.0)
    keep=vals>1e-10*mx
    if not np.any(keep):return np.zeros_like(g),0
    d=-(vecs[:,keep]@((vecs[:,keep].T@g)/vals[keep]))
    return d,int(keep.sum())

def run_one(payload):
    instance,graph,seed,graphs=payload;p=build_problem(instance,graph,graphs);theta=canonical_biased_theta(p,seed);initial=theta.copy();evals=0;first=-1;ming=math.inf;history=[];accepted=[];ranks=[]
    first_exact_cos=math.nan;first_target_cos=math.nan
    for step in range(STEPS):
        E,g,prob,I=evaluate(theta,p.F,p.C,True);evals+=1;gap=float(E-p.ground);ming=min(ming,gap);history.append(gap)
        if gap<=TOL:first=step;break
        d,rank=natural_direction(I,g);ranks.append(rank)
        limit=STEP_CAP*max(float(np.linalg.norm(p.c)),1e-12);dn=float(np.linalg.norm(d))
        if dn>limit:d*=limit/dn
        if step==0:first_exact_cos=1.0;first_target_cos=cosine(d,p.c)
        gd=float(g@d)
        if not np.isfinite(gd) or gd>=-1e-15:break
        alpha=1.0;accepted_alpha=0.0
        for _ in range(25):
            E2=evaluate(theta+alpha*d,p.F,p.C,False)[0];evals+=1
            if E2<=E+ARMIJO*alpha*gd:accepted_alpha=alpha;break
            alpha*=.5
        accepted.append(accepted_alpha)
        if accepted_alpha==0:break
        theta+=accepted_alpha*d
        if len(history)>=STAGNATION_WINDOW and max(history[-STAGNATION_WINDOW:])-min(history[-STAGNATION_WINDOW:])<STAGNATION_TOL:break
    E,g,prob,I=evaluate(theta,p.F,p.C,True);evals+=1;gap=float(E-p.ground);dom=int(np.argmax(prob));condition,rank,minv,maxv=eff_condition(I);beta=float(theta@p.c/max(p.c@p.c,1e-15));trans=theta-beta*p.c
    return {'instance_id':p.instance_id,'instance_width':p.instance_width,'split':p.split,'graph':graph,'seed':seed,'budget':STEPS,'success':int(first>=0),'first_success_step':first if first>=0 else math.nan,'minimum_gap':ming,'final_gap':gap,'final_pstar':float(prob[p.pidx]),'final_dominant_probability':float(prob[dom]),'final_dominant_gap':float(p.C[dom]-p.ground),'final_dominant_hamming':int(np.sum(p.bits[dom]!=np.asarray(p.planted))),'final_dominant_local_min':int(local_min(p.bits,p.C,dom)),'final_gradient_rms':float(np.sqrt(np.mean(g*g))),'final_fisher_condition':condition,'final_fisher_rank':rank,'gradient_evaluations':evals,'iterations':len(history),'mean_accepted_alpha':float(np.mean(accepted)) if accepted else math.nan,'minimum_accepted_alpha':float(np.min(accepted)) if accepted else math.nan,'first_direction_cosine_exact_natural':first_exact_cos,'first_direction_cosine_projected_target':first_target_cos,'final_target_beta':beta,'final_transverse_norm':float(np.linalg.norm(trans)),'initial_target_beta':float(initial@p.c/max(p.c@p.c,1e-15))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();fam=generate_family(INSTANCES);graphs=json.loads((GRAPHS/'partial_graphs.json').read_text());payload=[(inst,g,s,graphs) for inst in fam for g in ('chain','problem_tree','width2','width3') for s in SEEDS];start=time.time();rows=[]
    ctx=mp.get_context('fork')
    with ctx.Pool(a.workers,maxtasksperchild=4) as pool:
        for i,row in enumerate(pool.imap_unordered(run_one,payload),1):
            rows.append(row)
            if i%20==0:print('completed',i,'/',len(payload),flush=True)
    d=pd.DataFrame(rows).sort_values(['instance_id','graph','seed']);d.to_csv(RESULTS/'exact_natural_oracle.csv',index=False)
    agg=d.groupby(['split','graph'],as_index=False).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success_step=('first_success_step','mean'),median_minimum_gap=('minimum_gap','median'),mean_iterations=('iterations','mean'),mean_gradient_evaluations=('gradient_evaluations','mean'),trap_like_rate=('final_dominant_probability',lambda x:float(np.mean(np.asarray(x)>=.9))))
    agg.to_csv(RESULTS/'exact_natural_oracle_summary.csv',index=False);print(agg.to_string(index=False));print('elapsed',time.time()-start)
if __name__=='__main__':main()
