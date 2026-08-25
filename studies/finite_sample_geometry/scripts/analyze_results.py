from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'finite_sample_geometry'
B=30000;RNG=np.random.default_rng(20260824)

def cluster_rate(g):
    rates=g.groupby('instance_id').success.mean().to_numpy(float)
    draws=rates[RNG.integers(0,len(rates),size=(B,len(rates)))].mean(1)
    q=np.quantile(draws,[.025,.5,.975])
    return pd.Series({'success_rate':g.success.mean(),'ci_low':q[0],'ci_median':q[1],'ci_high':q[2],'instances':len(rates),'trajectories':len(g),'successes':int(g.success.sum())})

def paired_effect(a,b):
    idx=['instance_id','parameter_seed'];aa=a.set_index(idx).success.rename('a');bb=b.set_index(idx).success.rename('b');p=pd.concat([aa,bb],axis=1).dropna()
    inst=(p.a-p.b).groupby('instance_id').mean().to_numpy(float)
    draws=inst[RNG.integers(0,len(inst),size=(B,len(inst)))].mean(1);q=np.quantile(draws,[.025,.5,.975])
    return {'difference':float((p.a-p.b).mean()),'ci_low':q[0],'ci_median':q[1],'ci_high':q[2],'a_only':int(((p.a==1)&(p.b==0)).sum()),'b_only':int(((p.a==0)&(p.b==1)).sum()),'both':int(((p.a==1)&(p.b==1)).sum()),'neither':int(((p.a==0)&(p.b==0)).sum()),'pairs':len(p)}

def main():
    df=pd.read_csv(RESULTS/'finite_sample_broad_trajectories.csv');trap=pd.read_csv(RESULTS/'finite_sample_trap_replays.csv');ind=pd.read_csv(RESULTS/'independent_full_fisher_trajectories.csv')
    dims=df.groupby('instance_id').parameter_dimension.first().to_dict()
    groupcols=['split','sample_budget','method']
    agg=df.groupby(groupcols).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success=('first_success_step','mean'),median_first_success=('first_success_step','median'),mean_samples_consumed=('total_samples','mean'),median_samples_consumed=('total_samples','median'),total_samples_consumed=('total_samples','sum'),trap_count=('final_trap','sum'),trap_rate=('final_trap','mean'),mean_direction_cosine=('first_direction_cosine','mean'),mean_direction_relative_error=('first_direction_relative_error','mean'),mean_initial_rank=('first_sample_rank','mean'),mean_storage_entries=('stored_metric_entries','mean')).reset_index()
    agg['samples_per_observed_success']=agg.total_samples_consumed/agg.successes.replace(0,np.nan)
    agg.to_csv(RESULTS/'broad_aggregate_by_split.csv',index=False)
    agg[agg.split=='evaluation'].to_csv(RESULTS/'broad_aggregate_evaluation.csv',index=False)

    evaldf=df[df.split=='evaluation'].copy()
    ci=evaldf.groupby(['sample_budget','method'],group_keys=False).apply(cluster_rate,include_groups=False).reset_index()
    ci.to_csv(RESULTS/'evaluation_cluster_bootstrap_rates.csv',index=False)

    effects=[]
    for budget in (64,256,1024,4096):
        adam=evaldf[(evaldf.sample_budget==budget)&(evaldf.method=='sampled_adam')]
        for method in ['sampled_diagonal_fisher','sampled_two_block_fisher','sampled_full_fisher','ray_plus_residual']:
            other=evaldf[(evaldf.sample_budget==budget)&(evaldf.method==method)]
            effects.append({'sample_budget':budget,'method_a':method,'method_b':'sampled_adam',**paired_effect(other,adam)})
        analytic=evaldf[evaldf.method=='analytic_target_cooling']
        effects.append({'sample_budget':budget,'method_a':'analytic_target_cooling','method_b':'sampled_adam',**paired_effect(analytic,adam)})
    pd.DataFrame(effects).to_csv(RESULTS/'evaluation_paired_effects_vs_adam.csv',index=False)

    # Direction quality, rank, and storage.
    direction=evaldf.groupby(['sample_budget','method']).agg(trajectories=('success','size'),success_rate=('success','mean'),mean_rank=('first_sample_rank','mean'),mean_parameter_dimension=('parameter_dimension','mean'),mean_direction_cosine=('first_direction_cosine','mean'),median_direction_cosine=('first_direction_cosine','median'),mean_relative_error=('first_direction_relative_error','mean'),max_identity_residual=('maximum_sample_identity_residual','max'),mean_storage_entries=('stored_metric_entries','mean')).reset_index()
    direction['mean_rank_deficit']=direction.mean_parameter_dimension-direction.mean_rank
    direction.to_csv(RESULTS/'direction_rank_storage_summary.csv',index=False)

    full=evaldf[evaldf.method=='sampled_full_fisher'].copy();full['rank_deficit']=full.parameter_dimension-full.first_sample_rank
    full.groupby(['sample_budget','success']).agg(n=('success','size'),mean_rank=('first_sample_rank','mean'),mean_deficit=('rank_deficit','mean'),mean_cosine=('first_direction_cosine','mean'),mean_relative_error=('first_direction_relative_error','mean'),mean_final_gap=('final_gap','mean')).reset_index().to_csv(RESULTS/'full_fisher_rank_success.csv',index=False)

    # Budget pairings 64 versus 4096.
    budgetrows=[]
    for method in ['sampled_adam','sampled_diagonal_fisher','sampled_two_block_fisher','sampled_full_fisher','ray_plus_residual']:
        low=evaldf[(evaldf.method==method)&(evaldf.sample_budget==64)].set_index(['instance_id','parameter_seed']).success
        high=evaldf[(evaldf.method==method)&(evaldf.sample_budget==4096)].set_index(['instance_id','parameter_seed']).success
        p=pd.concat([low.rename('low'),high.rename('high')],axis=1)
        budgetrows.append({'method':method,'low_only_success':int(((p.low==1)&(p.high==0)).sum()),'high_only_success':int(((p.low==0)&(p.high==1)).sum()),'both_success':int(((p.low==1)&(p.high==1)).sum()),'neither_success':int(((p.low==0)&(p.high==0)).sum()),'rate_64':float(p.low.mean()),'rate_4096':float(p.high.mean())})
    pd.DataFrame(budgetrows).to_csv(RESULTS/'budget_64_vs_4096_paired.csv',index=False)

    # Independent-batch full-Fisher control (the total budget is split between gradient and Fisher batches).
    indagg=ind.groupby(['split','sample_budget']).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success=('first_success_step','mean'),mean_samples_consumed=('total_samples','mean'),total_samples_consumed=('total_samples','sum'),trap_count=('final_trap','sum'),trap_rate=('final_trap','mean'),mean_rank=('first_sample_rank','mean'),mean_direction_cosine=('first_direction_cosine','mean'),mean_gradient_fisher_mismatch=('first_gradient_fisher_mismatch','mean'),mean_storage_entries=('stored_metric_entries','mean')).reset_index()
    indagg['samples_per_observed_success']=indagg.total_samples_consumed/indagg.successes.replace(0,np.nan)
    indagg.to_csv(RESULTS/'independent_full_fisher_aggregate.csv',index=False)
    # Paired same-batch versus independent-batch full Fisher.
    samedf=evaldf[evaldf.method=='sampled_full_fisher']
    ie=ind[ind.split=='evaluation']
    comparisons=[]
    for budget in (64,256,1024,4096):
        a=samedf[samedf.sample_budget==budget];b=ie[ie.sample_budget==budget]
        comparisons.append({'sample_budget':budget,'method_a':'sampled_full_fisher_same_batch','method_b':'independent_full_fisher',**paired_effect(a,b)})
    pd.DataFrame(comparisons).to_csv(RESULTS/'same_vs_independent_full_fisher.csv',index=False)

    # Trap replay aggregation.
    trapagg=trap.groupby(['checkpoint','sample_budget','method']).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success=('first_success_step','mean'),median_minimum_gap=('minimum_gap','median'),trap_count=('final_trap','sum'),mean_initial_rank=('first_sample_rank','mean'),mean_direction_cosine=('first_direction_cosine','mean'),mean_samples=('total_samples','mean')).reset_index()
    trapagg.to_csv(RESULTS/'trap_replay_aggregate.csv',index=False)
    trapinst=trap.groupby(['instance_id','checkpoint','sample_budget','method']).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success=('first_success_step','mean'),mean_initial_rank=('first_sample_rank','mean'),mean_direction_cosine=('first_direction_cosine','mean'),median_final_gap=('final_gap','median')).reset_index()
    trapinst.to_csv(RESULTS/'trap_replay_by_instance.csv',index=False)

    summary={
      'broad_rows':len(df),'evaluation_rows':len(evaldf),'independent_rows':len(ind),'trap_rows':len(trap),
      'maximum_sample_identity_residual':float(df.maximum_sample_identity_residual.max()),
      'evaluation':{},'trap_replay':{}
    }
    for method in ['analytic_target_cooling','ray_plus_residual','sampled_full_fisher','sampled_two_block_fisher','sampled_diagonal_fisher','sampled_adam']:
      g=evaldf[evaldf.method==method]
      summary['evaluation'][method]={str(int(b)):{'success_rate':float(x.success.mean()),'successes':int(x.success.sum()),'trajectories':len(x),'mean_first_success':float(x.first_success_step.mean()) if x.success.any() else None,'samples_per_observed_success':float(x.total_samples.sum()/x.success.sum()) if x.success.sum() else None,'trap_rate':float(x.final_trap.mean())} for b,x in g.groupby('sample_budget')}
    summary['evaluation']['independent_full_fisher']={str(int(b)):{'success_rate':float(x.success.mean()),'successes':int(x.success.sum()),'trajectories':len(x),'mean_first_success':float(x.first_success_step.mean()) if x.success.any() else None,'samples_per_observed_success':float(x.total_samples.sum()/x.success.sum()) if x.success.sum() else None,'trap_rate':float(x.final_trap.mean()),'mean_gradient_fisher_mismatch':float(x.first_gradient_fisher_mismatch.mean())} for b,x in ind[ind.split=='evaluation'].groupby('sample_budget')}
    for checkpoint in (199,999):
      summary['trap_replay'][str(checkpoint)]={}
      for method in ['analytic_target_cooling','ray_plus_residual','sampled_full_fisher','sampled_adam']:
       g=trap[(trap.checkpoint==checkpoint)&(trap.method==method)]
       summary['trap_replay'][str(checkpoint)][method]={str(int(b)):{'success_rate':float(x.success.mean()),'successes':int(x.success.sum()),'trajectories':len(x),'mean_first_success':float(x.first_success_step.mean()) if x.success.any() else None,'mean_rank':float(x.first_sample_rank.mean())} for b,x in g.groupby('sample_budget')}
    (RESULTS/'analysis_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
