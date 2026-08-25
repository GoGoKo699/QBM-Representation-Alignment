from __future__ import annotations
import hashlib,math,pickle,sys,time
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];OUT=REPOSITORY/'results'/'finite_sample_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';BOUNDARY_RESULTS=REPOSITORY/'results'/'boundary_geometry';TASK=OUT/'independent_tasks';TASK.mkdir(exist_ok=True);sys.path.insert(0,str(REPOSITORY/'src'))
from qbm_alignment.certificate_family import generate_family
from qbm_alignment.optimizer_geometry import build,canonical_biased_theta,geom,trapped
from finite_sample_study import batch_prob,batch_grad,cap,BUDGETS,SEEDS,STEPS,TOL,storage

def seed(*x):return int.from_bytes(hashlib.sha256('|'.join(map(str,x)).encode()).digest()[:8],'little')
def sample_gradient(P,p,rng,M):
 idx=rng.choice(p.size,size=M,p=p);X=P.F[idx];y=P.C[idx];Xc=X-X.mean(0);yc=y-y.mean();return -(Xc.T@yc)/(M-1)
def sample_cov(P,p,rng,M):
 idx=rng.choice(p.size,size=M,p=p);X=P.F[idx];Xc=X-X.mean(0);I=Xc.T@Xc/(M-1);return I,int(np.linalg.matrix_rank(I,tol=1e-10))
def worker(payload):
 inst,M=payload;P=build(inst);K=len(SEEDS);T=np.column_stack([canonical_biased_theta(P,s) for s in SEEDS]);active=np.ones(K,bool);first=np.full(K,-1,int);ming=np.full(K,np.inf);igr=np.full(K,np.nan);batches=np.zeros(K,int);rank=np.full(K,-1,int);cos=np.full(K,np.nan);rel=np.full(K,np.nan);mismatch=np.zeros(K)
 rngg=[np.random.default_rng(seed('ind-g',inst.instance_id,s,M)) for s in SEEDS];rngi=[np.random.default_rng(seed('ind-I',inst.instance_id,s,M)) for s in SEEDS]
 Mg=M//2;Mi=M-Mg
 for step in range(STEPS):
  E,Q=batch_prob(P,T);gap=E-P.ground
  if step==0:
   G=batch_grad(P,Q,E);igr=np.sqrt(np.mean(G*G,1))
  ming=np.minimum(ming,gap);new=active&(gap<=TOL);first[new]=step;active[new]=False
  if not np.any(active) or step==STEPS-1:break
  for j in np.flatnonzero(active):
   g=sample_gradient(P,Q[:,j],rngg[j],Mg);I,r=sample_cov(P,Q[:,j],rngi[j],Mi);scale=max(float(np.trace(I)/I.shape[0]),1e-12);R=I+1e-10*scale*np.eye(I.shape[0])
   try:d=-np.linalg.solve(R,g)
   except np.linalg.LinAlgError:d=-np.linalg.lstsq(R,g,rcond=1e-10)[0]
   d=cap(d,P.c,.5);T[:,j]+=d;batches[j]+=1
   if step==0:
    rank[j]=r;dn=np.linalg.norm(d);cn=np.linalg.norm(P.c);cos[j]=d@P.c/(dn*cn) if dn else math.nan;ref=.5*P.c;rel[j]=np.linalg.norm(d-ref)/np.linalg.norm(ref);mismatch[j]=np.max(np.abs(g+I@P.c))
 E,Q=batch_prob(P,T);G=batch_grad(P,Q,E);gr=np.sqrt(np.mean(G*G,1));gap=E-P.ground;ps=Q[P.pidx];dom=np.argmax(Q,0);dp=Q[dom,np.arange(K)];ratio=gr/np.maximum(igr,1e-300);rows=[]
 for j,s in enumerate(SEEDS):
  candidate=gap[j]>.1 and dp[j]>=.9 and ps[j]<=.1 and ratio[j]<=.1;cond=math.nan;trap=0
  if candidate:
   gg=geom(P,T[:,j],float(igr[j]));cond=gg['fisher_condition'];trap=int(trapped(gg,P))
  rows.append({'instance_id':inst.instance_id,'width':inst.width,'split':'calibration' if inst.instance_id.endswith('_i1') else 'evaluation','parameter_seed':s,'sample_budget':M,'gradient_samples_per_update':Mg,'fisher_samples_per_update':Mi,'method':'independent_full_fisher','success':int(first[j]>=0),'first_success_step':first[j] if first[j]>=0 else math.nan,'minimum_gap':float(ming[j]),'final_gap':float(gap[j]),'final_pstar':float(ps[j]),'final_dominant_probability':float(dp[j]),'final_fisher_condition':cond,'final_gradient_rms_ratio':float(ratio[j]),'final_trap':trap,'sample_batches':int(batches[j]),'total_samples':int(batches[j]*M),'first_direction_cosine':float(cos[j]),'first_direction_relative_error':float(rel[j]),'first_sample_rank':int(rank[j]),'first_gradient_fisher_mismatch':float(mismatch[j]),'parameter_dimension':P.c.size,'stored_metric_entries':storage('sampled_full_fisher',P.c.size,P.instance.n)})
 return rows

def main():
 fam=generate_family(INSTANCES);tasks=[(i,M) for i in fam for M in BUDGETS];allrows=[]
 # serial with immediate task files, resumable
 for k,task in enumerate(tasks,1):
  inst,M=task;p=TASK/f'{inst.instance_id}_M{M}.pkl'
  if p.exists():rows=pickle.loads(p.read_bytes())
  else:rows=worker(task);p.write_bytes(pickle.dumps(rows,protocol=5))
  allrows.extend(rows);print(k,'/',len(tasks),flush=True)
 pd.DataFrame(allrows).to_csv(OUT/'independent_full_fisher_trajectories.csv',index=False)
if __name__=='__main__':main()
