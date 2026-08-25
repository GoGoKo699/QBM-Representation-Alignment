from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from confirmatory_common import Adam,ARMIJO,PARAMETER_SEEDS,PSEUDOINVERSE_CUT,STEPS,build_problem,cosine,effective_spectrum,exact_state,load_manifest,pseudoinverse_direction
BUDGET=1000;TOL=.1

def geom(problem,r,theta,initial_rms,method,step):
    E,g,p,I,_=exact_state(theta,problem,r,want_fisher=True);cond,rank,mn,mx=effective_spectrum(I);dom=int(np.argmax(p));c=r.coefficients;beta=float(theta@c/max(c@c,1e-15));trans=theta-beta*c
    return {'step':step,'method':method,'energy':E,'gap':E-problem.instance.ground_energy,'normalized_gap':(E-problem.instance.ground_energy)/problem.instance.spectral_gap,'pstar':float(p[problem.instance.ground_index]),'dominant_probability':float(p[dom]),'dominant_gap':float(problem.cost[dom]-problem.instance.ground_energy),'gradient_rms':float(np.sqrt(np.mean(g*g))),'gradient_ratio':float(np.sqrt(np.mean(g*g)))/max(initial_rms,1e-300),'fisher_condition':cond,'fisher_rank':rank,'target_cosine':cosine(theta,c),'transverse_norm':float(np.linalg.norm(trans)),'target_beta':beta}

def restarted_adam(problem,r,theta0):
    theta=theta0.copy();opt=Adam(len(theta));E,g,p,I,_=exact_state(theta,problem,r,True);initial=float(np.sqrt(np.mean(g*g)));logs=[];first=-1
    for step in range(BUDGET):
        row=geom(problem,r,theta,initial,'restarted_adam',step);logs.append(row)
        if row['normalized_gap']<=TOL:first=step;break
        E,g,p,I,_=exact_state(theta,problem,r,False);theta+=opt.step(g)
    return first,logs

def exact_natural(problem,r,theta0):
    theta=theta0.copy();E,g,p,I,_=exact_state(theta,problem,r,True);initial=float(np.sqrt(np.mean(g*g)));logs=[];first=-1;history=[]
    for step in range(BUDGET):
        E,g,p,I,_=exact_state(theta,problem,r,True);row=geom(problem,r,theta,initial,'exact_natural',step);logs.append(row);history.append(row['normalized_gap'])
        if row['normalized_gap']<=TOL:first=step;break
        d,_=pseudoinverse_direction(I,g);limit=.5*max(float(np.linalg.norm(r.coefficients)),1e-12);dn=float(np.linalg.norm(d));d=d*(limit/dn) if dn>limit else d;gd=float(g@d)
        if not np.isfinite(gd) or gd>=-1e-15:break
        alpha=1.;accepted=0.
        for _ in range(25):
            trial=exact_state(theta+alpha*d,problem,r,False)[0]
            if trial<=E+ARMIJO*alpha*gd:accepted=alpha;break
            alpha*=.5
        if accepted==0:break
        theta+=accepted*d
        if len(history)>=15 and max(history[-15:])-min(history[-15:])<1e-10:break
    return first,logs

def ray(problem,r,theta0):
    c=r.coefficients;theta=float(theta0@c/max(c@c,1e-15))*c;E,g,p,I,_=exact_state(theta,problem,r,True);initial=float(np.sqrt(np.mean(g*g)));logs=[];first=-1
    for step in range(BUDGET):
        row=geom(problem,r,theta,initial,'ray_projection_cooling',step);logs.append(row)
        if row['normalized_gap']<=TOL:first=step;break
        theta+=.02*c
    return first,logs

def main():
    instances={x.instance_id:x for x in load_manifest(ROOT/'instances'/'confirmatory_sparse_ising_manifest.json')};summary=pd.concat([pd.read_csv(p) for p in sorted((ROOT/'results'/'raw').glob('*_summary.csv'))],ignore_index=True);q=summary[(summary.graph=='full')&(summary.method=='adam')&(summary.initialization=='target_biased')&(summary.boundary_trap==1)];rows=[];logs=[]
    for x in q.itertuples(index=False):
        problem=build_problem(instances[x.instance_id]);r=problem.representations['full'];npz=np.load(ROOT/'results'/'states'/f'{x.instance_id}_states.npz');theta=npz['full__adam__target_biased__final'][list(PARAMETER_SEEDS).index(int(x.seed))]
        for name,fn in [('restarted_adam',restarted_adam),('exact_natural',exact_natural),('ray_projection_cooling',ray)]:
            first,lg=fn(problem,r,theta);rows.append({'instance_id':x.instance_id,'seed':int(x.seed),'source_step':int(x.final_step),'method':name,'success':int(first>=0),'first_success_step':first if first>=0 else math.nan,'minimum_normalized_gap':min(y['normalized_gap'] for y in lg),'final_normalized_gap':lg[-1]['normalized_gap'],'recorded_states':len(lg)});[y.update(instance_id=x.instance_id,seed=int(x.seed),source_step=int(x.final_step)) for y in lg];logs.extend(lg)
    out=ROOT/'results'/'replay';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'trap_replay_summary.csv',index=False);pd.DataFrame(logs).to_csv(out/'trap_replay_logs.csv.gz',index=False,compression='gzip');(out/'metadata.json').write_text(json.dumps({'qualifying_traps':len(q),'budget':BUDGET,'methods':['restarted_adam','exact_natural','ray_projection_cooling']},indent=2)+'\n');print('qualifying',len(q),'rows',len(rows))
if __name__=='__main__':main()
