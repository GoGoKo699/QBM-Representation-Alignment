from __future__ import annotations
import json,math,multiprocessing as mp,sys,time
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs'
sys.path.insert(0,str(STUDY/'scripts'));sys.path.insert(0,str(REPOSITORY/'src'))
from partial_alignment_study import SEEDS,STEPS,TOLERANCE,FULL_SHRINK,STEP_CAP,build_problem,canonical_biased_theta,exact_batch,regularized_solve,cap,stream_seed,generate_family,cosine,exact_natural_direction
from qbm_alignment.certificate_family import evaluate,eff_condition,local_min
TASKS=(('problem_tree',256),('width3',64),('width3',256))

def moments(problem,prob,rng,m):
    ia=rng.choice(prob.size,size=m,p=prob);ib=rng.choice(prob.size,size=m,p=prob)
    Xa=problem.F[ia];ya=problem.C[ia];ca=Xa-Xa.mean(0);cy=ya-ya.mean();g=-(ca.T@cy)/(m-1)
    Xb=problem.F[ib];cb=Xb-Xb.mean(0);I=cb.T@cb/(m-1);return g,I,int(np.linalg.matrix_rank(cb,tol=1e-10))

def run_group(payload):
    inst,graph,m,graphs=payload;p=build_problem(inst,graph,graphs);theta=np.column_stack([canonical_biased_theta(p,s) for s in SEEDS]);active=np.ones(len(SEEDS),bool);first=np.full(len(SEEDS),-1);ming=np.full(len(SEEDS),np.inf);batches=np.zeros(len(SEEDS),int);initg=np.full(len(SEEDS),np.nan);cosx=np.full(len(SEEDS),np.nan);rank0=np.full(len(SEEDS),-1);rngs=[np.random.default_rng(stream_seed('partial_equal',p.instance_id,p.graph,s,m)) for s in SEEDS]
    for step in range(STEPS):
        E,prob,gexact=exact_batch(p,theta);gap=E-p.ground
        if step==0:initg=np.sqrt(np.mean(gexact*gexact,axis=1))
        ming=np.minimum(ming,gap);new=active&(gap<=TOLERANCE);first[new]=step;active[new]=False
        if not np.any(active) or step==STEPS-1:break
        for i in np.flatnonzero(active):
            g,I,r=moments(p,prob[:,i],rngs[i],m);d=cap(regularized_solve(I,g,FULL_SHRINK),p.c,STEP_CAP);theta[:,i]+=d;batches[i]+=1
            if step==0:cosx[i]=cosine(d,exact_natural_direction(p,theta[:,i]-d));rank0[i]=r
    E,prob,g=exact_batch(p,theta);gap=E-p.ground;dom=np.argmax(prob,axis=0);gr=np.sqrt(np.mean(g*g,axis=1));rows=[]
    for i,s in enumerate(SEEDS):
        cand=bool(gap[i]>TOLERANCE and p.C[dom[i]]>p.ground+1e-12 and prob[dom[i],i]>=.9 and prob[p.pidx,i]<=.1 and gr[i]/max(initg[i],1e-300)<=.1)
        rows.append({'instance_id':p.instance_id,'instance_width':p.instance_width,'split':p.split,'graph':graph,'samples_per_batch':m,'total_samples_per_update':2*m,'seed':s,'method':'independent_equal_full_fisher','success':int(first[i]>=0),'first_success_step':int(first[i]) if first[i]>=0 else math.nan,'minimum_gap':float(ming[i]),'sample_batches':int(batches[i]),'total_samples':int(batches[i]*2*m),'first_direction_cosine_exact_ng':float(cosx[i]),'first_sample_rank':int(rank0[i]),'final_gap':float(gap[i]),'final_pstar':float(prob[p.pidx,i]),'final_dominant_probability':float(prob[dom[i],i]),'final_gradient_ratio':float(gr[i]/max(initg[i],1e-300)),'final_trap_candidate':int(cand)})
    return rows

def main():
    start=time.time();fam=generate_family(INSTANCES);graphs=json.loads((GRAPHS/'partial_graphs.json').read_text());payload=[(i,g,m,graphs) for i in fam for g,m in TASKS];rows=[];ctx=mp.get_context('fork')
    with ctx.Pool(8,maxtasksperchild=4) as pool:
        for k,out in enumerate(pool.imap_unordered(run_group,payload),1):rows.extend(out);print('completed',k,'/',len(payload),flush=True)
    d=pd.DataFrame(rows).sort_values(['instance_id','graph','samples_per_batch','seed']);d.to_csv(RESULTS/'independent_equal_full_fisher.csv',index=False)
    a=d.groupby(['split','graph','samples_per_batch','total_samples_per_update'],as_index=False).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success_step=('first_success_step','mean'),median_minimum_gap=('minimum_gap','median'),trap_candidate_rate=('final_trap_candidate','mean'),total_samples=('total_samples','sum'),mean_first_direction_cosine=('first_direction_cosine_exact_ng','mean'),mean_first_rank=('first_sample_rank','mean'))
    a.to_csv(RESULTS/'independent_equal_full_fisher_summary.csv',index=False);print(a.to_string(index=False));print('elapsed',time.time()-start)
if __name__=='__main__':main()
