from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results';ANALYSIS=RESULTS/'analysis';ANALYSIS.mkdir(parents=True,exist_ok=True)
PRIMARY=(('H1','adam','target_biased','problem_tree','chain',.15),('H2','adam','target_biased','problem_tree','random_tree',.10),('H3','exact_natural','target_biased','problem_tree','chain',.25))
B=20000;SEED=20260825;MATERIAL=.5

def holm(p):
    ordered=sorted(p,key=p.get);out={};running=0.;m=len(ordered)
    for rank,key in enumerate(ordered):
        running=max(running,min(1.,(m-rank)*p[key]));out[key]=running
    return out,ordered

def paired(data,opt,init,treat,control,rng):
    d=data[(data.method==opt)&(data.initialization==init)]
    table=d.pivot_table(index=['instance_id','seed'],columns='graph',values='success',aggfunc='first')[[treat,control]].dropna()
    inst=(table[treat]-table[control]).groupby(level=0).mean();v=inst.to_numpy(float)
    draws=v[rng.integers(0,len(v),size=(B,len(v)))].mean(1)
    raw=np.quantile(draws,[.025,.975]);alpha=.05/3;bonf=np.quantile(draws,[alpha/2,1-alpha/2])
    pl=(np.sum(draws<=0)+1)/(B+1);pr=(np.sum(draws>=0)+1)/(B+1);pv=min(1.,2*min(pl,pr))
    return {'instances':len(v),'paired_trajectories':len(table),'point_difference':v.mean(),'raw_ci_low':raw[0],'raw_ci_high':raw[1],'bonferroni_ci_low':bonf[0],'bonferroni_ci_high':bonf[1],'raw_bootstrap_p':pv,'treatment_only_wins':int(((table[treat]==1)&(table[control]==0)).sum()),'control_only_wins':int(((table[treat]==0)&(table[control]==1)).sum()),'both_success':int(((table[treat]==1)&(table[control]==1)).sum()),'neither_success':int(((table[treat]==0)&(table[control]==0)).sum()),'instance_differences':' '.join(f'{x:.6g}' for x in v)},draws

def main():
    summaries=pd.concat([pd.read_csv(p) for p in sorted((RESULTS/'raw').glob('*_summary.csv'))],ignore_index=True)
    resources=pd.read_csv(RESULTS/'confirmatory_preparation_resources.csv')
    summaries.to_csv(ANALYSIS/'confirmatory_trajectory_summary.csv',index=False)
    aggregate=summaries.groupby(['method','initialization','graph'],as_index=False).agg(trajectories=('success','size'),successes=('success','sum'),success_rate=('success','mean'),mean_first_success_step=('first_success_step','mean'),median_minimum_normalized_gap=('minimum_normalized_gap','median'),trap_count=('boundary_trap','sum'))
    aggregate.to_csv(ANALYSIS/'confirmatory_aggregate_summary.csv',index=False)
    rng=np.random.default_rng(SEED);rows=[];draws={};pvals={};thresholds={}
    for h,opt,init,treat,control,threshold in PRIMARY:
        row,samples=paired(summaries,opt,init,treat,control,rng);row.update(hypothesis=h,optimizer=opt,initialization=init,treatment=treat,control=control,practical_threshold=threshold);rows.append(row);draws[h]=samples;pvals[h]=row['raw_bootstrap_p'];thresholds[h]=threshold
    adjusted,ordered=holm(pvals);rank={h:i for i,h in enumerate(ordered)};m=len(PRIMARY)
    for row in rows:
        h=row['hypothesis'];alpha=.05/(m-rank[h]);lo,hi=np.quantile(draws[h],[alpha/2,1-alpha/2]);row['holm_adjusted_p']=adjusted[h];row['holm_stepdown_ci_low']=lo;row['holm_stepdown_ci_high']=hi;row['passes_effect_threshold']=int(row['point_difference']>=thresholds[h]);row['holm_interval_excludes_zero']=int(lo>0);row['primary_pass']=int(row['passes_effect_threshold'] and row['holm_interval_excludes_zero'])
    effects=pd.DataFrame(rows).sort_values('hypothesis');effects.to_csv(ANALYSIS/'confirmatory_primary_effects.csv',index=False)
    problem=resources[resources.graph=='problem_tree'];full=resources[resources.graph=='full'];pair=problem.merge(full,on='instance_id',suffixes=('_problem_tree','_full'));pair['angle_ratio']=pair.conditional_angle_entries_problem_tree/pair.conditional_angle_entries_full;pair['cnot_ratio']=pair.gray_code_cnot_upper_bound_problem_tree/pair.gray_code_cnot_upper_bound_full;pair.to_csv(ANALYSIS/'confirmatory_resource_pairs.csv',index=False)
    g4=bool((pair.width_problem_tree==1).all() and (pair.width_full>=2).all() and pair.angle_ratio.median()<=MATERIAL and pair.cnot_ratio.median()<=MATERIAL)
    passes={r.hypothesis:bool(r.primary_pass) for r in effects.itertuples(index=False)}
    trap_count=int(summaries[(summaries.graph=='full')&(summaries.method=='adam')&(summaries.initialization=='target_biased')].boundary_trap.sum())
    if all(passes.values()) and g4:decision='FULL_GO'
    elif passes.get('H1',False) and (effects.point_difference>=0).all():decision='NARROW_PAPER'
    elif trap_count>0:decision='MECHANISM_ONLY_CANDIDATE'
    else:decision='NO_GO_REFRAME'
    outcome={'decision':decision,'primary_pass':passes,'preparation_G4_pass':g4,'material_resource_ratio_threshold':MATERIAL,'median_angle_ratio_problem_tree_to_full':float(pair.angle_ratio.median()),'median_cnot_ratio_problem_tree_to_full':float(pair.cnot_ratio.median()),'full_target_biased_adam_traps':trap_count,'bootstrap_replicates':B,'bootstrap_seed':SEED,'holm_correction':True,'bonferroni_simultaneous_intervals_reported':True}
    (ANALYSIS/'confirmatory_go_no_go.json').write_text(json.dumps(outcome,indent=2,sort_keys=True)+'\n')
    print(aggregate.to_string(index=False));print(effects.to_string(index=False));print(json.dumps(outcome,indent=2,sort_keys=True))
if __name__=='__main__':main()
