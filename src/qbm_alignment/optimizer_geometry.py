from __future__ import annotations
import json, math, multiprocessing as mp, pickle, os, itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from .certificate_family import (
    Instance,
    generate_family,
    state_features,
    costs,
    coefficients,
    evaluate,
    eff_condition,
    local_min,
)

TRAP = {
    "energy_gap_strictly_greater_than": 0.1,
    "dominant_state_probability_at_least": 0.9,
    "planted_state_probability_at_most": 0.1,
    "effective_fisher_condition_number_at_least": 1_000_000.0,
    "gradient_rms_at_most_fraction_of_initial": 0.1,
}
SEEDS=(0,19,42,50,101); TOL=.1; LR=.02; B1=.9; B2=.999; EPS=1e-8

@dataclass
class Problem:
    instance:Instance; bits:np.ndarray; F:np.ndarray; C:np.ndarray; c:np.ndarray; ground:float; pidx:int

def build(inst:Instance)->Problem:
    bits,F=state_features(inst.n,inst.edges);C=costs(bits,inst.clauses);c=coefficients(inst.n,inst.clauses,inst.edges);pidx=int(''.join(map(str,inst.planted)),2)
    return Problem(inst,bits,F,C,c,float(C.min()),pidx)

class Adam:
    def __init__(self,d):self.m=np.zeros(d);self.v=np.zeros(d);self.t=0
    def step(self,g):
        self.t+=1;self.m=B1*self.m+(1-B1)*g;self.v=B2*self.v+(1-B2)*g*g
        return -LR*(self.m/(1-B1**self.t))/(np.sqrt(self.v/(1-B2**self.t))+EPS)

def cos(a,b):
    d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(a@b/d) if d else math.nan

def geom(P:Problem,theta,initial_grad,update=None):
    E,g,p,I=evaluate(theta,P.F,P.C,True);dom=int(np.argmax(p));gr=float(np.sqrt(np.mean(g*g)));cond,rank,mn,mx=eff_condition(I)
    beta=float(theta@P.c/(P.c@P.c));u=theta-beta*P.c;H=float(-np.sum(p[p>0]*np.log(p[p>0])))
    row={'energy':E,'gap':E-P.ground,'pstar':float(p[P.pidx]),'dominant_index':dom,'dominant_probability':float(p[dom]),'dominant_gap':float(P.C[dom]-P.ground),'dominant_hamming':int(np.sum(P.bits[dom]!=np.asarray(P.instance.planted))),'dominant_local_min':local_min(P.bits,P.C,dom),'entropy':H,'effective_support':float(np.exp(H)),'gradient_rms':gr,'gradient_ratio':gr/max(initial_grad,1e-300),'fisher_condition':cond,'fisher_rank':rank,'fisher_min_kept':mn,'fisher_max':mx,'theta_norm':float(np.linalg.norm(theta)),'theta_target_cosine':cos(theta,P.c),'beta_projection':beta,'transverse_norm':float(np.linalg.norm(u)),'transverse_ratio':float(np.linalg.norm(u)/max(abs(beta)*np.linalg.norm(P.c),1e-300))}
    if update is not None:
        row.update(update_norm=float(np.linalg.norm(update)),update_target_cosine=cos(update,P.c),gradient_update_dot=float(g@update))
        Ic=I@P.c;den=float(np.sqrt(max(update@I@update,0))*np.sqrt(max(P.c@Ic,0)));row['update_fisher_target_cosine']=float(update@Ic/den) if den else math.nan
    return row

def trapped(m,P):
    return bool(m['gap']>TRAP['energy_gap_strictly_greater_than'] and P.C[int(m['dominant_index'])]>P.ground+1e-12 and m['dominant_probability']>=TRAP['dominant_state_probability_at_least'] and m['pstar']<=TRAP['planted_state_probability_at_most'] and m['fisher_condition']>=TRAP['effective_fisher_condition_number_at_least'] and m['gradient_ratio']<=TRAP['gradient_rms_at_most_fraction_of_initial'])

def armijo(P,theta,g,d,alpha0=1.0):
    if float(g@d)>=0:return np.zeros_like(theta),0.,0
    E=evaluate(theta,P.F,P.C,False)[0];a=alpha0
    for k in range(25):
        if evaluate(theta+a*d,P.F,P.C,False)[0]<=E+1e-4*a*float(g@d):return a*d,a,k+1
        a*=.5
    return np.zeros_like(theta),0.,25

def diag_direction(g,p,F):
    mu=p@F;diag=np.maximum(1.0-mu*mu,0.0);return -g/(diag+1e-6*max(float(diag.max()),1.))

def damped_direction(g,I):
    I=(I+I.T)/2;mx=max(float(np.linalg.eigvalsh(I)[-1]),1.);return np.linalg.solve(I+1e-8*mx*np.eye(len(g)),-g)

def baseline(inst:Instance,steps=1000):
    P=build(inst);theta=P.c.copy();opt=Adam(len(theta));E,g,p,_=evaluate(theta,P.F,P.C,False);g0=float(np.sqrt(np.mean(g*g)));first=-1;ming=math.inf;firsttrap=-1;logs=[];states={}
    checkpoints={0,25,50,100,150,199,399,599,799,999}
    for t in range(steps):
        E,g,p,_=evaluate(theta,P.F,P.C,False);gap=E-P.ground;ming=min(ming,gap)
        if first<0 and gap<=TOL:first=t
        upd=opt.step(g) if t<steps-1 else np.zeros_like(theta)
        dom=int(np.argmax(p));ratio=float(np.sqrt(np.mean(g*g))/g0);candidate=firsttrap<0 and gap>.1 and p[dom]>=.9 and p[P.pidx]<=.1 and ratio<=.1
        if t in checkpoints or candidate:
            m=geom(P,theta,g0,upd);is_t=trapped(m,P)
            if is_t and firsttrap<0:firsttrap=t
            logs.append({'instance_id':inst.instance_id,'width':inst.width,'step':t,'trap':is_t,**m})
        if t in checkpoints:states[t]=theta.copy()
        if t<steps-1:theta+=upd
    return {'instance_id':inst.instance_id,'width':inst.width,'success':first>=0,'first_success':first,'minimum_gap':ming,'first_trap':firsttrap,'final_gap':float(evaluate(theta,P.F,P.C,False)[0]-P.ground)},logs,states

def baseline_worker(inst):return baseline(inst)

def run_baselines(fam):
    # Only the two documented long-run failures require 1,000-state reproduction.
    targets=[x for x in fam if x.instance_id in {'ct_w5_i1','ct_w6_i1'}]
    with mp.get_context('fork').Pool(2) as pool:outs=pool.map(baseline_worker,targets)
    s=pd.DataFrame([x[0] for x in outs]);l=pd.DataFrame([r for x in outs for r in x[1]]);states={x[0]['instance_id']:x[2] for x in outs}
    return s,l,states

def intervention(P,theta0,method,steps=500,log_every=25):
    theta=theta0.copy()
    projection_displacement=0.0
    if method=='ray_projected':
        beta=float(theta@P.c/(P.c@P.c));projected=beta*P.c;projection_displacement=float(np.linalg.norm(theta-projected));theta=projected
    ad=Adam(len(theta)) if method in {'adam','projected_adam'} else None;E,g,p,_=evaluate(theta,P.F,P.C,False);g0=float(np.sqrt(np.mean(g*g)));first=-1;ming=math.inf;evals=0;logs=[]
    for t in range(steps):
        needI=method=='damped_natural' or t%log_every==0 or t==steps-1
        E,g,p,I=evaluate(theta,P.F,P.C,needI);evals+=1;gap=E-P.ground;ming=min(ming,gap)
        if first<0 and gap<=TOL:first=t
        upd=np.zeros_like(theta);alpha=math.nan;trials=0
        if t<steps-1 and first<0:
            if method=='adam':upd=ad.step(g)
            elif method=='armijo_gd':upd,alpha,trials=armijo(P,theta,g,-g);evals+=trials
            elif method in {'target_direction','ray_projected'}:upd=LR*P.c;alpha=LR
            elif method=='projected_adam':
                raw=ad.step(g);upd=(raw@P.c/(P.c@P.c))*P.c
            elif method=='exact_natural':upd,alpha,trials=armijo(P,theta,g,P.c);evals+=trials
            elif method=='diagonal_fisher':upd,alpha,trials=armijo(P,theta,g,diag_direction(g,p,P.F));evals+=trials
            elif method=='damped_natural':upd,alpha,trials=armijo(P,theta,g,damped_direction(g,I));evals+=trials
            else:raise ValueError(method)
        if t%log_every==0 or t==steps-1 or first==t:
            m=geom(P,theta,g0,upd);logs.append({'instance_id':P.instance.instance_id,'width':P.instance.width,'optimizer':method,'step':t,'success_at_step':first==t,'alpha':alpha,'line_trials':trials,'projection_displacement':projection_displacement,**m})
        if first>=0:break
        theta+=upd
    return {'instance_id':P.instance.instance_id,'width':P.instance.width,'optimizer':method,'success':first>=0,'first_success':first,'minimum_gap':ming,'final_gap':float(evaluate(theta,P.F,P.C,False)[0]-P.ground),'gradient_evaluations':evals,'projection_displacement':projection_displacement},logs

def replay_worker(payload):
    inst,checkpoint,theta=payload;P=build(inst);methods=('adam','armijo_gd','target_direction','ray_projected','projected_adam','exact_natural','damped_natural','diagonal_fisher');ss=[];ll=[]
    for m in methods:
        budget=500 if m=='damped_natural' else 1000
        s,l=intervention(P,theta,m,budget,25);s['replay_checkpoint']=checkpoint;s['replay_budget']=budget;ss.append(s)
        for r in l:r['replay_checkpoint']=checkpoint;ll.append(r)
    return ss,ll

def run_replays(fam,states):
    im={x.instance_id:x for x in fam};payload=[]
    for iid in states:
        for cp in (199,999):payload.append((im[iid],cp,states[iid][cp]))
    with mp.get_context('fork').Pool(2) as pool:outs=pool.map(replay_worker,payload)
    return pd.DataFrame([r for x in outs for r in x[0]]),pd.DataFrame([r for x in outs for r in x[1]])


def canonical_biased_theta(P,seed):
    pairs=tuple(itertools.combinations(range(P.instance.n),2));index={edge:i for i,edge in enumerate(pairs)}
    full=.3*np.random.default_rng(seed).standard_normal(P.instance.n+len(pairs))
    active=np.asarray(list(range(P.instance.n))+[P.instance.n+index[e] for e in P.instance.edges],dtype=np.int64)
    return P.c+full[active]

def suite_worker(payload):
    inst,init,seed=payload;P=build(inst);theta=P.c.copy() if init=='exact_target' else canonical_biased_theta(P,seed);methods=('adam','armijo_gd','target_direction','ray_projected','projected_adam','exact_natural','diagonal_fisher');ss=[];ll=[]
    for m in methods:
        s,l=intervention(P,theta,m,200,50);s.update(initialization=init,seed=seed);ss.append(s)
        for r in l:r.update(initialization=init,seed=seed);ll.append(r)
    return ss,ll

def run_suite(fam):
    payload=[]
    for x in fam:
        payload.append((x,'exact_target',-1))
        for seed in SEEDS:payload.append((x,'target_biased',seed))
    with mp.get_context('fork').Pool(6) as pool:outs=list(pool.imap_unordered(suite_worker,payload))
    return pd.DataFrame([r for x in outs for r in x[0]]),pd.DataFrame([r for x in outs for r in x[1]])
