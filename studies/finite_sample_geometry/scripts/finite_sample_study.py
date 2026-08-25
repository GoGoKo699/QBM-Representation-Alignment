from __future__ import annotations
import hashlib,json,math,multiprocessing as mp,pickle,sys,time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];OUT=REPOSITORY/'results'/'finite_sample_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';BOUNDARY_RESULTS=REPOSITORY/'results'/'boundary_geometry';sys.path.insert(0,str(REPOSITORY/'src'))
from qbm_alignment.certificate_family import evaluate,generate_family
from qbm_alignment.optimizer_geometry import Adam,build,canonical_biased_theta,geom,trapped
BUDGETS=(64,256,1024,4096);SEEDS=(0,19,42,50,101)
METHODS=('sampled_adam','sampled_diagonal_fisher','sampled_two_block_fisher','sampled_full_fisher','ray_plus_residual')
STEPS=200;TOL=.1
FULL_STEP=.5;BLOCK_SHRINK=.1;BLOCK_RIDGE=1e-3;DIAG_RIDGE=1e-3;RAY_BETA=.5;RAY_KEEP=.95;RAY_RESID=.1
LOG_STEPS={0,1,2,5,10,25,50,100,150,199}
@dataclass(frozen=True)
class Moments:
 g:np.ndarray;diag:np.ndarray;Xc:np.ndarray;I:np.ndarray|None;rank:int;resid:float

def sseed(*x):return int.from_bytes(hashlib.sha256('|'.join(map(str,x)).encode()).digest()[:8],'little')
def batch_prob(P,T):
 S=P.F@T;L=-S;L-=L.max(0,keepdims=True);W=np.exp(np.clip(L,-745,0));Q=W/W.sum(0,keepdims=True);return P.C@Q,Q
def batch_grad(P,Q,E):return -((Q*P.C[:,None]).T@P.F-E[:,None]*(Q.T@P.F))
def moments(P,p,rng,M,full=False,diagnostics=False):
 idx=rng.choice(p.size,size=M,p=p);X=P.F[idx];y=P.C[idx];Xc=X-X.mean(0);yc=y-y.mean();g=-(Xc.T@yc)/(M-1);diag=np.sum(Xc*Xc,0)/(M-1);I=Xc.T@Xc/(M-1) if full else None
 if diagnostics:
  resid=float(np.max(np.abs(g+Xc.T@(Xc@P.c)/(M-1))));rank=int(np.linalg.matrix_rank(I if I is not None else Xc,tol=1e-10))
 else:resid=0.;rank=-1
 return Moments(g,diag,Xc,I,rank,resid)
def cap(d,c,f):
 n=float(np.linalg.norm(d));lim=f*float(np.linalg.norm(c));return d*(lim/n) if n>lim and n>0 else d
def block_dir(P,m,g):
 d=np.zeros_like(g);n=P.instance.n
 for sl in (slice(0,n),slice(n,g.size)):
  X=m.Xc[:,sl];I=X.T@X/(X.shape[0]-1);q=np.diag(I);scale=max(float(q.mean()),1e-12);R=(1-BLOCK_SHRINK)*I+BLOCK_SHRINK*np.diag(q)+BLOCK_RIDGE*scale*np.eye(I.shape[0])
  try:d[sl]=-np.linalg.solve(R,g[sl])
  except np.linalg.LinAlgError:d[sl]=-np.linalg.lstsq(R,g[sl],rcond=1e-10)[0]
 return d
def storage(method,d,n):
 p=d-n
 return {'sampled_adam':2*d,'sampled_diagonal_fisher':d,'sampled_two_block_fisher':n*(n+1)//2+p*(p+1)//2,'ray_plus_residual':n*(n+1)//2+p*(p+1)//2,'sampled_full_fisher':d*(d+1)//2,'analytic_target_cooling':d}[method]
def update(P,theta,method,p,rng,M,adam,diagnostics):
 if method=='analytic_target_cooling':return RAY_BETA*P.c,(-1,0.,1.,0.)
 m=moments(P,p,rng,M,method=='sampled_full_fisher',diagnostics);g=m.g
 if method=='sampled_adam':d=adam.step(g)
 elif method=='sampled_diagonal_fisher':d=cap(-g/(m.diag+DIAG_RIDGE*max(float(m.diag.max()),1e-12)),P.c,FULL_STEP)
 elif method=='sampled_two_block_fisher':d=cap(block_dir(P,m,g),P.c,FULL_STEP)
 elif method=='sampled_full_fisher':
  scale=max(float(np.trace(m.I)/m.I.shape[0]),1e-12);R=m.I+1e-10*scale*np.eye(m.I.shape[0])
  try:d=-np.linalg.solve(R,g)
  except np.linalg.LinAlgError:d=-np.linalg.lstsq(R,g,rcond=1e-10)[0]
  d=cap(d,P.c,FULL_STEP)
 elif method=='ray_plus_residual':
  c2=float(P.c@P.c);beta=float(theta@P.c/c2);u=theta-beta*P.c;gp=g-float(g@P.c/c2)*P.c;dr=block_dir(P,m,gp);dr-=float(dr@P.c/c2)*P.c;dr=cap(dr,P.c,RAY_RESID);new=(beta+RAY_BETA)*P.c+RAY_KEEP*u+dr;d=new-theta
 else:raise ValueError(method)
 dn=float(np.linalg.norm(d));cn=float(np.linalg.norm(P.c));cos=float(d@P.c/(dn*cn)) if dn else math.nan;ref=RAY_BETA*P.c;rel=float(np.linalg.norm(d-ref)/np.linalg.norm(ref));return d,(m.rank,m.resid,cos,rel)
def run_group(P,specs,steps):
 K=len(specs);d=P.c.size;T=np.column_stack([s['theta0'] for s in specs]);active=np.ones(K,bool);first=np.full(K,-1,int);ming=np.full(K,np.inf);igr=np.full(K,np.nan);batches=np.zeros(K,int);mxres=np.zeros(K);frank=np.full(K,-1,int);fcos=np.full(K,np.nan);frel=np.full(K,np.nan);adams=[Adam(d) if s['method']=='sampled_adam' else None for s in specs];rngs=[np.random.default_rng(int(s['stream_seed'])) for s in specs];logs=[]
 for step in range(steps):
  E,Q=batch_prob(P,T);gap=E-P.ground;ps=Q[P.pidx]
  if step==0:
   G=batch_grad(P,Q,E);igr=np.sqrt(np.mean(G*G,1))
  ming=np.minimum(ming,gap);new=active&(gap<=TOL);first[new]=step;active[new]=False
  if step in LOG_STEPS or np.any(new):
   for j,s in enumerate(specs):
    if step in LOG_STEPS or new[j]:
     beta=float(T[:,j]@P.c/(P.c@P.c));u=T[:,j]-beta*P.c;logs.append({**{k:v for k,v in s.items() if k!='theta0'},'step':step,'gap':float(gap[j]),'pstar':float(ps[j]),'target_beta':beta,'transverse_norm':float(np.linalg.norm(u))})
  if not np.any(active) or step==steps-1:break
  for j in np.flatnonzero(active):
   s=specs[j];M=int(s['sample_budget']);dlt,diag=update(P,T[:,j],str(s['method']),Q[:,j],rngs[j],M,adams[j],step==0);T[:,j]+=dlt
   if s['method']!='analytic_target_cooling':batches[j]+=1;mxres[j]=max(mxres[j],diag[1])
   if step==0:frank[j]=diag[0];fcos[j]=diag[2];frel[j]=diag[3]
 E,Q=batch_prob(P,T);G=batch_grad(P,Q,E);gr=np.sqrt(np.mean(G*G,1));gap=E-P.ground;ps=Q[P.pidx];dom=np.argmax(Q,0);dp=Q[dom,np.arange(K)];ratio=gr/np.maximum(igr,1e-300);cfg=json.loads((REPOSITORY/'data'/'trap_definition.json').read_text())['conditions'];rows=[]
 for j,s in enumerate(specs):
  cand=bool(gap[j]>cfg['energy_gap_strictly_greater_than'] and P.C[int(dom[j])]>P.ground+1e-12 and dp[j]>=cfg['dominant_state_probability_at_least'] and ps[j]<=cfg['planted_state_probability_at_most'] and ratio[j]<=cfg['gradient_rms_at_most_fraction_of_initial']);cond=math.nan;trap=0
  if cand:
   gg=geom(P,T[:,j],float(igr[j]));cond=float(gg['fisher_condition']);trap=int(trapped(gg,P))
  method=str(s['method']);M=int(s['sample_budget']);rows.append({**{k:v for k,v in s.items() if k!='theta0'},'success':int(first[j]>=0),'first_success_step':first[j] if first[j]>=0 else math.nan,'minimum_gap':float(ming[j]),'final_gap':float(gap[j]),'final_pstar':float(ps[j]),'final_dominant_probability':float(dp[j]),'final_fisher_condition':cond,'final_gradient_rms_ratio':float(ratio[j]),'final_trap':trap,'sample_batches':int(batches[j]),'total_samples':int(batches[j]*M),'first_direction_cosine':float(fcos[j]),'first_direction_relative_error':float(frel[j]),'first_sample_rank':int(frank[j]),'maximum_sample_identity_residual':float(mxres[j]),'parameter_dimension':d,'stored_metric_entries':storage(method,d,P.instance.n)})
 return rows,logs
def broad_worker(payload):
 inst,M=payload;P=build(inst);split='calibration' if inst.instance_id.endswith('_i1') else 'evaluation';specs=[]
 if M==0:
  for seed in SEEDS:specs.append({'instance_id':inst.instance_id,'width':inst.width,'split':split,'parameter_seed':seed,'sample_budget':0,'method':'analytic_target_cooling','stream_seed':0,'theta0':canonical_biased_theta(P,seed)})
 else:
  for seed in SEEDS:
   stream=sseed('broad',inst.instance_id,seed,M)
   for method in METHODS:specs.append({'instance_id':inst.instance_id,'width':inst.width,'split':split,'parameter_seed':seed,'sample_budget':M,'method':method,'stream_seed':stream,'theta0':canonical_biased_theta(P,seed)})
 return run_group(P,specs,STEPS)
def trap_worker(payload):
 iid,cp,M=payload;fam=generate_family(INSTANCES);im={x.instance_id:x for x in fam};P=build(im[iid]);state=pickle.load((BOUNDARY_RESULTS/'baseline_states.pkl').open('rb'))[iid][cp];specs=[]
 if M==64:specs.append({'instance_id':iid,'checkpoint':cp,'sample_replicate':-1,'sample_budget':0,'method':'analytic_target_cooling','stream_seed':0,'theta0':state})
 for rep in range(5):
  stream=sseed('trap',iid,cp,M,rep)
  for method in ('sampled_adam','sampled_full_fisher','ray_plus_residual'):specs.append({'instance_id':iid,'checkpoint':cp,'sample_replicate':rep,'sample_budget':M,'method':method,'stream_seed':stream,'theta0':state})
 return run_group(P,specs,STEPS)
def main():
 start=time.time();fam=generate_family(INSTANCES);payload=[(i,b) for i in fam for b in (0,)+BUDGETS];outs=[]
 with mp.get_context('fork').Pool(5) as pool:
  for k,o in enumerate(pool.imap_unordered(broad_worker,payload),1):outs.append(o);print('broad',k,'/',len(payload),flush=True)
 pd.DataFrame([r for o in outs for r in o[0]]).to_csv(OUT/'finite_sample_broad_trajectories.csv',index=False);pd.DataFrame([r for o in outs for r in o[1]]).to_csv(OUT/'finite_sample_broad_logs.csv.gz',index=False,compression='gzip')
 tp=[(iid,cp,M) for iid in ('ct_w5_i1','ct_w6_i1') for cp in (199,999) for M in BUDGETS];tout=[]
 with mp.get_context('fork').Pool(5) as pool:
  for k,o in enumerate(pool.imap_unordered(trap_worker,tp),1):tout.append(o);print('trap',k,'/',len(tp),flush=True)
 pd.DataFrame([r for o in tout for r in o[0]]).to_csv(OUT/'finite_sample_trap_replays.csv',index=False);pd.DataFrame([r for o in tout for r in o[1]]).to_csv(OUT/'finite_sample_trap_logs.csv.gz',index=False,compression='gzip')
 meta={'budgets':BUDGETS,'parameter_seeds':SEEDS,'methods':METHODS+('analytic_target_cooling',),'steps':STEPS,'calibration_instances':[f'ct_w{w}_i1' for w in (3,4,5,6)],'evaluation_instances':[f'ct_w{w}_i{i}' for w in (3,4,5,6) for i in range(2,6)],'trap_replicates':5,'elapsed_seconds':time.time()-start};(OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
