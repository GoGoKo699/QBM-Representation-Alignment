from __future__ import annotations
import json,math,multiprocessing as mp,sys,time
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs'
sys.path.insert(0,str(STUDY/'scripts'));sys.path.insert(0,str(REPOSITORY/'src'))
from partial_alignment_study import SEEDS,STEPS,TOLERANCE,FULL_SHRINK,RIDGE,STEP_CAP,build_problem,canonical_biased_theta,exact_batch,regularized_solve,cap,stream_seed,generate_family,cosine,exact_natural_direction
from qbm_alignment.certificate_family import evaluate,eff_condition,local_min

TASKS=(('problem_tree',256),('width3',64),('width3',256),('width3',1024))

def independent_moments(problem,probability,rng,total):
    a=total//2;b=total-a
    ia=rng.choice(probability.size,size=a,p=probability);ib=rng.choice(probability.size,size=b,p=probability)
    Xa=problem.F[ia];ya=problem.C[ia];ca=Xa-Xa.mean(0);cy=ya-ya.mean();g=-(ca.T@cy)/(a-1)
    Xb=problem.F[ib];cb=Xb-Xb.mean(0);I=cb.T@cb/(b-1);rank=int(np.linalg.matrix_rank(cb,tol=1e-10))
    return g,I,rank

def run_group(payload):
    instance,graph,budget,graphs=payload;p=build_problem(instance,graph,graphs);count=len(SEEDS);theta=np.column_stack([canonical_biased_theta(p,s) for s in SEEDS]);active=np.ones(count,bool);first=np.full(count,-1);ming=np.full(count,np.inf);sample_batches=np.zeros(count,int);initial_grad=np.full(count,np.nan);first_cos=np.full(count,np.nan);first_target=np.full(count,np.nan);first_rank=np.full(count,-1);rngs=[np.random.default_rng(stream_seed('partial',p.instance_id,p.graph,s,budget)) for s in SEEDS]
    for step in range(STEPS):
        E,prob,gexact=exact_batch(p,theta);gap=E-p.ground
        if step==0:initial_grad=np.sqrt(np.mean(gexact*gexact,axis=1))
        ming=np.minimum(ming,gap);new=active&(gap<=TOLERANCE);first[new]=step;active[new]=False
        if not np.any(active) or step==STEPS-1:break
        for i in np.flatnonzero(active):
            g,I,rank=independent_moments(p,prob[:,i],rngs[i],budget);d=cap(regularized_solve(I,g,FULL_SHRINK),p.c,STEP_CAP);theta[:,i]+=d;sample_batches[i]+=1
            if step==0:
                exact=exact_natural_direction(p,theta[:,i]-d);first_cos[i]=cosine(d,exact);first_target[i]=cosine(d,p.c);first_rank[i]=rank
    E,prob,g=exact_batch(p,theta);gap=E-p.ground;dom=np.argmax(prob,axis=0);grad=np.sqrt(np.mean(g*g,axis=1));rows=[]
    for i,seed in enumerate(SEEDS):
        candidate=bool(gap[i]>TOLERANCE and p.C[dom[i]]>p.ground+1e-12 and prob[dom[i],i]>=.9 and prob[p.pidx,i]<=.1 and grad[i]/max(initial_grad[i],1e-300)<=.1)
        cond=math.nan;rank=-1
        if candidate:
            _E,_g,_p,I=evaluate(theta[:,i],p.F,p.C,True);cond,rank,_mn,_mx=eff_condition(I)
        rows.append({'instance_id':p.instance_id,'instance_width':p.instance_width,'split':p.split,'graph':graph,'sample_budget':budget,'seed':seed,'method':'independent_full_fisher','parameter_dimension':p.F.shape[1],'success':int(first[i]>=0),'first_success_step':int(first[i]) if first[i]>=0 else math.nan,'minimum_gap':float(ming[i]),'initial_gradient_rms':float(initial_grad[i]),'sample_batches':int(sample_batches[i]),'total_samples':int(sample_batches[i]*budget),'first_direction_cosine_exact_ng':float(first_cos[i]),'first_direction_cosine_projected_target':float(first_target[i]),'first_sample_rank':int(first_rank[i]),'final_gap':float(gap[i]),'final_pstar':float(prob[p.pidx,i]),'final_dominant_probability':float(prob[dom[i],i]),'final_dominant_gap':float(p.C[dom[i]]-p.ground),'final_dominant_hamming':int(np.sum(p.bits[dom[i]]!=np.asarray(p.planted))),'final_dominant_local_min':int(local_min(p.bits,p.C,int(dom[i]))),'final_gradient_rms':float(grad[i]),'final_gradient_ratio':float(grad[i]/max(initial_grad[i],1e-300)),'final_fisher_condition':float(cond),'final_fisher_rank':int(rank),'final_trap_candidate':int(candidate)})
    return rows

def main():
    start=time.time();fam=generate_family(INSTANCES);graphs=json.loads((GRAPHS/'partial_graphs.json').read_text());payload=[(inst,g,b,graphs) for inst in fam for g,b in TASKS];rows=[];ctx=mp.get_context('fork')
    with ctx.Pool(8,maxtasksperchild=4) as pool:
        for i,out in enumerate(pool.imap_unordered(run_group,payload),1):rows.extend(out);print('completed',i,'/',len(payload),flush=True)
    d=pd.DataFrame(rows).sort_values(['instance_id','graph','sample_budget','seed']);d.to_csv(RESULTS/'independent_full_fisher.csv',index=False)
    agg=d.groupby(['split','graph','sample_budget'],as_index=False).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success_step=('first_success_step','mean'),median_minimum_gap=('minimum_gap','median'),trap_candidate_rate=('final_trap_candidate','mean'),total_samples=('total_samples','sum'),mean_first_direction_cosine=('first_direction_cosine_exact_ng','mean'),mean_first_rank=('first_sample_rank','mean'))
    agg.to_csv(RESULTS/'independent_full_fisher_summary.csv',index=False);print(agg.to_string(index=False));print('elapsed',time.time()-start)
if __name__=='__main__':main()
