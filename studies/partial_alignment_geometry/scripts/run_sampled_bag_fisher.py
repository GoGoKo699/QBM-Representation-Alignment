from __future__ import annotations
import json,math,multiprocessing as mp,sys,time
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs';sys.path.insert(0,str(REPOSITORY/'src'));sys.path.insert(0,str(STUDY/'scripts'))
from partial_alignment_study import SEEDS,STEPS,TOLERANCE,SHRINK,STEP_CAP,build_problem,canonical_biased_theta,exact_batch,sample_moments,regularized_solve,cap,stream_seed,generate_family,cosine,exact_natural_direction
from qbm_alignment.certificate_family import evaluate,eff_condition,local_min
BUDGET=256

def bag_blocks(problem,payload):
    order=tuple(payload[problem.instance_id][problem.graph]['order']);adj={i:set() for i in range(problem.n)}
    for a,b in problem.edges:adj[a].add(b);adj[b].add(a)
    rem=set(range(problem.n));supports=[{i} for i in range(problem.n)]+[set(e) for e in problem.edges];blocks=[]
    for v in order:
        nb=sorted(adj[v]&rem);bag=set([v,*nb])
        for i,a in enumerate(nb):
            for b in nb[i+1:]:adj[a].add(b);adj[b].add(a)
        rem.remove(v);idx=tuple(i for i,s in enumerate(supports) if s<=bag)
        if idx:blocks.append(idx)
    mult=np.zeros(problem.F.shape[1])
    for b in blocks:mult[list(b)]+=1
    if np.any(mult==0):raise RuntimeError('bag blocks do not cover features')
    return tuple(blocks),mult

def bag_direction(problem,moments,g,blocks,mult):
    d=np.zeros_like(g)
    for block in blocks:
        idx=np.asarray(block,int);X=moments.centered[:,idx];I=X.T@X/(X.shape[0]-1);d[idx]+=regularized_solve(I,g[idx],SHRINK)
    return cap(d/mult,problem.c,STEP_CAP)

def run_group(payload):
    inst,graph,graphs=payload;p=build_problem(inst,graph,graphs);blocks,mult=bag_blocks(p,graphs);storage=sum(len(b)*(len(b)+1)//2 for b in blocks);theta=np.column_stack([canonical_biased_theta(p,s) for s in SEEDS]);active=np.ones(len(SEEDS),bool);first=np.full(len(SEEDS),-1);ming=np.full(len(SEEDS),np.inf);batches=np.zeros(len(SEEDS),int);initg=np.full(len(SEEDS),np.nan);cosx=np.full(len(SEEDS),np.nan);targetcos=np.full(len(SEEDS),np.nan);rank0=np.full(len(SEEDS),-1);rngs=[np.random.default_rng(stream_seed('partial',p.instance_id,p.graph,s,BUDGET)) for s in SEEDS]
    for step in range(STEPS):
        E,prob,gexact=exact_batch(p,theta);gap=E-p.ground
        if step==0:initg=np.sqrt(np.mean(gexact*gexact,axis=1))
        ming=np.minimum(ming,gap);new=active&(gap<=TOLERANCE);first[new]=step;active[new]=False
        if not np.any(active) or step==STEPS-1:break
        for i in np.flatnonzero(active):
            mom=sample_moments(p,prob[:,i],rngs[i],BUDGET,False,step==0);d=bag_direction(p,mom,mom.gradient,blocks,mult);theta[:,i]+=d;batches[i]+=1
            if step==0:cosx[i]=cosine(d,exact_natural_direction(p,theta[:,i]-d));targetcos[i]=cosine(d,p.c);rank0[i]=mom.sample_rank
    E,prob,g=exact_batch(p,theta);gap=E-p.ground;dom=np.argmax(prob,axis=0);gr=np.sqrt(np.mean(g*g,axis=1));rows=[]
    for i,s in enumerate(SEEDS):
        cand=bool(gap[i]>TOLERANCE and p.C[dom[i]]>p.ground+1e-12 and prob[dom[i],i]>=.9 and prob[p.pidx,i]<=.1 and gr[i]/max(initg[i],1e-300)<=.1)
        cond=math.nan;rank=-1
        if cand:
            _E,_g,_p,I=evaluate(theta[:,i],p.F,p.C,True);cond,rank,_mn,_mx=eff_condition(I)
        rows.append({'instance_id':p.instance_id,'instance_width':p.instance_width,'split':p.split,'graph':graph,'sample_budget':BUDGET,'seed':s,'method':'sampled_bag_fisher','parameter_dimension':p.F.shape[1],'success':int(first[i]>=0),'first_success_step':int(first[i]) if first[i]>=0 else math.nan,'minimum_gap':float(ming[i]),'initial_gradient_rms':float(initg[i]),'sample_batches':int(batches[i]),'total_samples':int(batches[i]*BUDGET),'stored_metric_entries':storage,'max_bag_feature_block':max(map(len,blocks)),'first_direction_cosine_exact_ng':float(cosx[i]),'first_direction_cosine_projected_target':float(targetcos[i]),'first_sample_rank':int(rank0[i]),'final_gap':float(gap[i]),'final_pstar':float(prob[p.pidx,i]),'final_dominant_probability':float(prob[dom[i],i]),'final_dominant_gap':float(p.C[dom[i]]-p.ground),'final_dominant_hamming':int(np.sum(p.bits[dom[i]]!=np.asarray(p.planted))),'final_dominant_local_min':int(local_min(p.bits,p.C,int(dom[i]))),'final_gradient_rms':float(gr[i]),'final_gradient_ratio':float(gr[i]/max(initg[i],1e-300)),'final_fisher_condition':float(cond),'final_fisher_rank':int(rank),'final_trap_candidate':int(cand)})
    return rows

def main():
    start=time.time();fam=generate_family(INSTANCES);graphs=json.loads((GRAPHS/'partial_graphs.json').read_text());payload=[(i,g,graphs) for i in fam for g in ('chain','problem_tree','width2','width3')];rows=[];ctx=mp.get_context('fork')
    with ctx.Pool(8,maxtasksperchild=4) as pool:
        for k,out in enumerate(pool.imap_unordered(run_group,payload),1):rows.extend(out);print('completed',k,'/',len(payload),flush=True)
    d=pd.DataFrame(rows).sort_values(['instance_id','graph','seed']);d.to_csv(RESULTS/'sampled_bag_fisher.csv',index=False)
    a=d.groupby(['split','graph'],as_index=False).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success_step=('first_success_step','mean'),median_minimum_gap=('minimum_gap','median'),trap_candidate_rate=('final_trap_candidate','mean'),total_samples=('total_samples','sum'),samples_per_observed_success=('total_samples',lambda x:math.nan),mean_first_direction_cosine=('first_direction_cosine_exact_ng','mean'),mean_storage=('stored_metric_entries','mean'),max_block=('max_bag_feature_block','max'))
    for idx,row in a.iterrows():a.loc[idx,'samples_per_observed_success']=row.total_samples/row.successes if row.successes else math.inf
    a.to_csv(RESULTS/'sampled_bag_fisher_summary.csv',index=False);print(a.to_string(index=False));print('elapsed',time.time()-start)
if __name__=='__main__':main()
