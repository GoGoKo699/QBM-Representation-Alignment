from __future__ import annotations
import sys
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];R=REPOSITORY/'results'/'finite_sample_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';BOUNDARY_RESULTS=REPOSITORY/'results'/'boundary_geometry';sys.path.insert(0,str(REPOSITORY/'src'))
from qbm_alignment.certificate_family import generate_family,evaluate
from qbm_alignment.optimizer_geometry import build,canonical_biased_theta

def check(cond,msg):
 if not cond:raise AssertionError(msg)

def main():
 broad=pd.read_csv(R/'finite_sample_broad_trajectories.csv');ind=pd.read_csv(R/'independent_full_fisher_trajectories.csv');trap=pd.read_csv(R/'finite_sample_trap_replays.csv')
 check(len(broad)==2100,f'broad rows {len(broad)}');check(len(ind)==400,f'ind rows {len(ind)}');check(len(trap)==244,f'trap rows {len(trap)}')
 expected={'analytic_target_cooling':100,'sampled_adam':400,'sampled_diagonal_fisher':400,'sampled_two_block_fisher':400,'sampled_full_fisher':400,'ray_plus_residual':400}
 check(broad.method.value_counts().to_dict()==expected,'method counts')
 check(not broad.duplicated(['instance_id','parameter_seed','sample_budget','method']).any(),'broad duplicates')
 check(not ind.duplicated(['instance_id','parameter_seed','sample_budget']).any(),'independent duplicates')
 check(float(broad.maximum_sample_identity_residual.max())<1e-12,'same-batch identity')
 evaldf=broad[broad.split=='evaluation']
 target={
  ('analytic_target_cooling',0):80,
  ('ray_plus_residual',64):80,('ray_plus_residual',256):80,('ray_plus_residual',1024):80,('ray_plus_residual',4096):80,
  ('sampled_full_fisher',64):73,('sampled_full_fisher',256):79,('sampled_full_fisher',1024):80,('sampled_full_fisher',4096):80,
  ('sampled_two_block_fisher',64):48,('sampled_two_block_fisher',256):57,('sampled_two_block_fisher',1024):58,('sampled_two_block_fisher',4096):58,
  ('sampled_diagonal_fisher',64):34,('sampled_diagonal_fisher',256):37,('sampled_diagonal_fisher',1024):36,('sampled_diagonal_fisher',4096):37,
  ('sampled_adam',64):33,('sampled_adam',256):35,('sampled_adam',1024):34,('sampled_adam',4096):34,
 }
 for (method,budget),count in target.items():
  got=int(evaldf[(evaldf.method==method)&(evaldf.sample_budget==budget)].success.sum());check(got==count,f'{method} {budget}: {got}')
 itarget={64:12,256:19,1024:67,4096:80}
 for budget,count in itarget.items():
  got=int(ind[(ind.split=='evaluation')&(ind.sample_budget==budget)].success.sum());check(got==count,f'ind {budget}: {got}')
 check(int(trap[trap.method=='ray_plus_residual'].success.sum())==80,'ray trap rescue')
 check(int(trap[trap.method=='sampled_adam'].success.sum())==0,'adam trap rescue')
 check(int(trap[trap.method=='sampled_full_fisher'].success.sum())==0,'full trap rescue')
 check(int(trap[trap.method=='analytic_target_cooling'].success.sum())==4,'analytic trap rescue')
 ranks=evaldf[evaldf.method=='sampled_full_fisher'].groupby('sample_budget').first_sample_rank.mean().to_numpy();check(np.all(np.diff(ranks)>0),'rank monotonicity')
 # Independent numerical check of ghat = -Ihat c from one finite batch.
 fam=generate_family(INSTANCES);P=build(fam[0]);theta=canonical_biased_theta(P,42);_,_,prob,_=evaluate(theta,P.F,P.C,False);rng=np.random.default_rng(20260824);idx=rng.choice(prob.size,size=256,p=prob);X=P.F[idx];y=P.C[idx];Xc=X-X.mean(0);yc=y-y.mean();I=Xc.T@Xc/255;g=-(Xc.T@yc)/255;res=float(np.max(np.abs(g+I@P.c)));check(res<1e-12,f'identity residual {res}')
 for name in ['evaluation_success_vs_samples','samples_per_observed_success','direction_alignment_vs_samples','full_fisher_rank_recovery','same_vs_independent_full_fisher','storage_success_pareto','trap_replay_success','trap_sample_rank','broad_trap_rate']:
  for suffix in ('.png','.pdf'):
   p=STUDY/'figures'/f'{name}{suffix}';check(p.exists() and p.stat().st_size>1000,str(p))
 report={'status':'PASS','broad_trajectories':len(broad),'independent_batch_controls':len(ind),'trap_replays':len(trap),'finite_sample_identity_residual':res}
 (R/'validation.json').write_text(__import__('json').dumps(report,indent=2)+'\n')
 print('Finite-sample geometry study validation passed.')
 print('  broad trajectories:',len(broad))
 print('  independent-batch controls:',len(ind))
 print('  trap replays:',len(trap))
 print('  finite-sample identity residual:',res)
if __name__=='__main__':main()
