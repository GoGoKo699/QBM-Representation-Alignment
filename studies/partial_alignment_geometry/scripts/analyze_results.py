from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];R=REPOSITORY/'results'/'partial_alignment_geometry'
LABELS={
 'sampled_adam':'Sampled Adam','sampled_diagonal_fisher':'Diagonal Fisher','sampled_two_block_fisher':'Two-block Fisher','sampled_star_fisher':'Graph-star Fisher','sampled_bag_fisher':'Elimination-bag Fisher','sampled_full_fisher':'Same-batch full Fisher','protected_ray_star':'Protected target + star','exact_natural_oracle':'Exact natural oracle','independent_full_fisher':'Independent split full Fisher','independent_equal_full_fisher':'Independent equal-batch full Fisher'}

def cluster_ci(df, value='success',B=30000,seed=20260824):
    rng=np.random.default_rng(seed);rates=df.groupby('instance_id')[value].mean().to_numpy(float);idx=rng.integers(0,len(rates),size=(B,len(rates)));x=rates[idx].mean(1);return np.quantile(x,[.025,.5,.975])

def paired_effect(left,right,B=30000,seed=20260824):
    keys=['instance_id','seed'];a=left.set_index(keys).success.rename('a');b=right.set_index(keys).success.rename('b');p=pd.concat([a,b],axis=1).dropna();by=(p.a-p.b).groupby(level=0).mean().to_numpy(float);rng=np.random.default_rng(seed);idx=rng.integers(0,len(by),size=(B,len(by)));x=by[idx].mean(1);q=np.quantile(x,[.025,.5,.975]);return {'paired_trajectories':len(p),'mean_difference':float((p.a-p.b).mean()),'ci_low':q[0],'ci_median':q[1],'ci_high':q[2],'left_only':int(((p.a==1)&(p.b==0)).sum()),'right_only':int(((p.a==0)&(p.b==1)).sum())}

def primary():
    base=pd.read_csv(R/'partial_alignment_trajectories.csv');base=base[(base.split=='evaluation')&(base.sample_budget==256)].copy()
    bag=pd.read_csv(R/'sampled_bag_fisher.csv');bag=bag[bag.split=='evaluation'].copy()
    oracle=pd.read_csv(R/'exact_natural_oracle.csv');oracle=oracle[oracle.split=='evaluation'].copy();oracle['method']='exact_natural_oracle';oracle['sample_budget']=0;oracle['total_samples']=0;oracle['stored_metric_entries']=oracle.apply(lambda r:int(r.final_fisher_rank*(r.final_fisher_rank+1)//2),axis=1);oracle['final_trap_candidate']=((oracle.success==0)&(oracle.final_dominant_gap>0)&(oracle.final_dominant_probability>=.9)&(oracle.final_pstar<=.1)&(oracle.final_gradient_rms<=1e-8)).astype(int)
    bag['sample_budget']=256
    allrows=pd.concat([base,bag,oracle],ignore_index=True,sort=False)
    rows=[]
    for (graph,method),g in allrows.groupby(['graph','method']):
        q=cluster_ci(g);succ=int(g.success.sum());samples=int(g.total_samples.fillna(0).sum())
        rows.append({'graph':graph,'method':method,'method_label':LABELS.get(method,method),'trajectories':len(g),'successes':succ,'success_rate':g.success.mean(),'cluster_ci_low':q[0],'cluster_ci_median':q[1],'cluster_ci_high':q[2],'mean_first_success_step':g.first_success_step.mean(),'median_minimum_gap':g.minimum_gap.median(),'trap_candidate_rate':g.final_trap_candidate.fillna(0).mean(),'total_samples':samples,'samples_per_observed_success':samples/succ if succ else math.inf,'median_stored_metric_entries':g.stored_metric_entries.median() if 'stored_metric_entries' in g else math.nan})
    out=pd.DataFrame(rows);out.to_csv(R/'primary_summary.csv',index=False)
    return allrows,out

def oracle_effects(allrows):
    rows=[]
    for graph in ('chain','problem_tree','width2','width3'):
        o=allrows[(allrows.graph==graph)&(allrows.method=='exact_natural_oracle')]
        for method in ('sampled_adam','sampled_diagonal_fisher','sampled_two_block_fisher','sampled_star_fisher','sampled_bag_fisher','sampled_full_fisher','protected_ray_star'):
            m=allrows[(allrows.graph==graph)&(allrows.method==method)]
            e=paired_effect(o,m);rows.append({'graph':graph,'oracle':'exact_natural_oracle','comparison_method':method,**e})
    d=pd.DataFrame(rows);d.to_csv(R/'oracle_effects.csv',index=False);return d

def batch_controls():
    same=pd.read_csv(R/'partial_alignment_trajectories.csv');same=same[(same.split=='evaluation')&(same.method=='sampled_full_fisher')]
    split=pd.read_csv(R/'independent_full_fisher.csv');split=split[split.split=='evaluation']
    equal=pd.read_csv(R/'independent_equal_full_fisher.csv');equal=equal[equal.split=='evaluation']
    rows=[]
    specs=[('problem_tree',256),('width3',64),('width3',256),('width3',1024)]
    for graph,budget in specs:
        s=same[(same.graph==graph)&(same.sample_budget==budget)]
        if len(s):
            q=cluster_ci(s);rows.append({'graph':graph,'nominal_budget':budget,'estimator':'same_batch','samples_per_update':budget,'gradient_batch':budget,'fisher_batch':budget,'trajectories':len(s),'successes':int(s.success.sum()),'success_rate':s.success.mean(),'ci_low':q[0],'ci_high':q[2],'mean_first_direction_cosine':s.first_direction_cosine_exact_ng.mean(),'mean_rank':s.first_sample_rank.mean()})
        u=split[(split.graph==graph)&(split.sample_budget==budget)]
        if len(u):
            q=cluster_ci(u);rows.append({'graph':graph,'nominal_budget':budget,'estimator':'independent_split_total','samples_per_update':budget,'gradient_batch':budget//2,'fisher_batch':budget-budget//2,'trajectories':len(u),'successes':int(u.success.sum()),'success_rate':u.success.mean(),'ci_low':q[0],'ci_high':q[2],'mean_first_direction_cosine':u.first_direction_cosine_exact_ng.mean(),'mean_rank':u.first_sample_rank.mean()})
        v=equal[(equal.graph==graph)&(equal.samples_per_batch==budget)]
        if len(v):
            q=cluster_ci(v);rows.append({'graph':graph,'nominal_budget':budget,'estimator':'independent_equal_batches','samples_per_update':2*budget,'gradient_batch':budget,'fisher_batch':budget,'trajectories':len(v),'successes':int(v.success.sum()),'success_rate':v.success.mean(),'ci_low':q[0],'ci_high':q[2],'mean_first_direction_cosine':v.first_direction_cosine_exact_ng.mean(),'mean_rank':v.first_sample_rank.mean()})
    d=pd.DataFrame(rows);d.to_csv(R/'full_fisher_batch_controls.csv',index=False)
    effects=[]
    for graph,budget in specs:
        s=same[(same.graph==graph)&(same.sample_budget==budget)]
        for label,u in [('independent_split_total',split[(split.graph==graph)&(split.sample_budget==budget)]),('independent_equal_batches',equal[(equal.graph==graph)&(equal.samples_per_batch==budget)])]:
            if len(s) and len(u):effects.append({'graph':graph,'budget':budget,'comparison':f'same_minus_{label}',**paired_effect(s,u)})
    pd.DataFrame(effects).to_csv(R/'batch_paired_effects.csv',index=False);return d

def diagnostics():
    first=pd.read_csv(R/'first_step_diagnostics.csv');first=first[first.split=='evaluation']
    fsum=first.groupby(['graph','method'],as_index=False).agg(n=('instance_id','size'),mean_cosine_exact_natural=('direction_cosine_exact_natural','mean'),median_cosine_exact_natural=('direction_cosine_exact_natural','median'),median_relative_error=('direction_relative_error_exact_natural','median'),mean_cosine_projected_target=('direction_cosine_projected_target','mean'),mean_sample_rank=('sample_rank','mean'),mean_partial_alignment_residual=('sample_alignment_residual','mean'))
    fsum.to_csv(R/'sampled_first_step_summary.csv',index=False)
    exact_raw=pd.read_csv(R/'exact_preconditioner_diagnostics.csv')
    exact_eval=exact_raw[exact_raw.split=='evaluation']
    exact=exact_eval.groupby(['graph','method'],as_index=False).agg(
        n=('instance_id','size'),
        cos_exact_mean=('direction_cosine_exact_natural','mean'),
        cos_exact_median=('direction_cosine_exact_natural','median'),
        relative_error_median=('direction_relative_error_exact_natural','median'),
        cos_target_mean=('direction_cosine_projected_target','mean'),
    )
    exact.to_csv(R/'exact_preconditioner_summary.csv',index=False)
    bag_raw=pd.read_csv(R/'bag_fisher_diagnostic.csv')
    bag_eval=bag_raw[bag_raw.split=='evaluation']
    bag=bag_eval.groupby('graph',as_index=False).agg(
        n=('instance_id','size'),
        cos_mean=('cosine_exact_natural','mean'),
        cos_median=('cosine_exact_natural','median'),
        relerr_median=('relative_error','median'),
        storage_median=('stored_entries','median'),
        maxblock=('max_block_size','max'),
    )
    bag.to_csv(R/'bag_diagnostic_summary.csv',index=False)

def main():
    allrows,summary=primary();oe=oracle_effects(allrows);bc=batch_controls();diagnostics()
    chain=pd.read_csv(R/'chain_representability_control.csv');chain.to_csv(R/'chain_representability.csv',index=False)
    metrics=pd.read_csv(R/'partial_graph_metrics_enriched.csv');metrics.to_csv(R/'graph_metrics.csv',index=False)
    data={'heldout_primary_trajectories':int(len(allrows)),'heldout_instances':16,'parameter_seeds':5,'graphs':['chain','problem_tree','width2','width3'],'main_sample_budget':256,'exact_oracle_success':{g:float(summary[(summary.graph==g)&(summary.method=='exact_natural_oracle')].success_rate.iloc[0]) for g in ['chain','problem_tree','width2','width3']},'same_batch_full_success':{g:float(summary[(summary.graph==g)&(summary.method=='sampled_full_fisher')].success_rate.iloc[0]) for g in ['chain','problem_tree','width2','width3']},'bag_fisher_exploratory':True}
    (R/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(summary[['graph','method_label','success_rate','cluster_ci_low','cluster_ci_high','mean_first_success_step','trap_candidate_rate','samples_per_observed_success']].to_string(index=False))
if __name__=='__main__':main()
