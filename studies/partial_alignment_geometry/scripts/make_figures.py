from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault('SOURCE_DATE_EPOCH','1787616000')
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['pdf.fonttype']=42;matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];R=REPOSITORY/'results'/'partial_alignment_geometry';F=STUDY/'figures';F.mkdir(exist_ok=True)
PDF={'Creator':'QBM partial-alignment study','Producer':'Matplotlib','CreationDate':None,'ModDate':None}

def save(fig,name):
    fig.tight_layout();fig.savefig(F/f'{name}.pdf',bbox_inches='tight',metadata=PDF);fig.savefig(F/f'{name}.png',dpi=220,bbox_inches='tight',metadata={'Software':'QBM representation-alignment project'});plt.close(fig)

# Main success by graph.
d=pd.read_csv(R/'primary_summary.csv')
methods=['sampled_adam','sampled_two_block_fisher','sampled_bag_fisher','sampled_full_fisher','protected_ray_star','exact_natural_oracle']
labels={'sampled_adam':'Adam','sampled_two_block_fisher':'Two-block','sampled_bag_fisher':'Bag Fisher','sampled_full_fisher':'Full Fisher','protected_ray_star':'Protected target','exact_natural_oracle':'Exact NG oracle'}
graphs=['chain','problem_tree','width2','width3'];gl={'chain':'Chain','problem_tree':'Problem tree','width2':'Width 2','width3':'Width 3'}
fig,axes=plt.subplots(1,4,figsize=(14.4,4.2),sharey=True)
for ax,g in zip(axes,graphs):
    sub=d[(d.graph==g)&(d.method.isin(methods))].set_index('method').reindex(methods);x=np.arange(len(methods));y=sub.success_rate.to_numpy();lo=y-sub.cluster_ci_low.to_numpy();hi=sub.cluster_ci_high.to_numpy()-y
    ax.bar(x,y);ax.errorbar(x,y,yerr=np.vstack([lo,hi]),fmt='none',capsize=3)
    ax.set_title(gl[g]);ax.set_xticks(x,labels=[labels[m] for m in methods],rotation=70,ha='right');ax.set_ylim(0,1.05);ax.grid(axis='y',alpha=.25)
axes[0].set_ylabel('Held-out success fraction')
fig.suptitle('Partial alignment: 256-sample methods versus exact natural-gradient oracle')
save(fig,'success_by_graph')

# Oracle ceiling and full Fisher gap.
fig,ax=plt.subplots(figsize=(7.2,4.5));x=np.arange(4);w=.28
for j,m in enumerate(['exact_natural_oracle','sampled_full_fisher','protected_ray_star']):
    sub=d[d.method==m].set_index('graph').loc[graphs];ax.bar(x+(j-1)*w,sub.success_rate,width=w,label=labels[m])
ax.set_xticks(x,labels=[gl[g] for g in graphs]);ax.set_ylim(0,1.05);ax.set_ylabel('Success fraction');ax.set_title('Exact geometric ceiling and finite-sample gap');ax.grid(axis='y',alpha=.25);ax.legend()
save(fig,'oracle_and_sampled_ceiling')

# Full Fisher batching controls.
b=pd.read_csv(R/'full_fisher_batch_controls.csv')
fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),sharey=True)
pt=b[(b.graph=='problem_tree')&(b.nominal_budget==256)]
axes[0].bar(np.arange(len(pt)),pt.success_rate);axes[0].set_xticks(np.arange(len(pt)),labels=['Same\nbatch','Split total','Independent\nequal'],rotation=0);axes[0].set_title('Problem tree, nominal budget 256');axes[0].set_ylabel('Success fraction');axes[0].set_ylim(0,1.05);axes[0].grid(axis='y',alpha=.25)
w3=b[b.graph=='width3']
for est,label in [('same_batch','Same batch'),('independent_split_total','Independent, split total'),('independent_equal_batches','Independent, equal batches')]:
    s=w3[w3.estimator==est].sort_values('nominal_budget');axes[1].plot(s.nominal_budget,s.success_rate,marker='o',label=label)
axes[1].set_xscale('log',base=2);axes[1].set_xticks([64,256,1024],labels=['64','256','1024']);axes[1].set_title('Width-3 sample scaling');axes[1].set_xlabel('Nominal samples');axes[1].grid(alpha=.25);axes[1].legend(fontsize=8)
fig.suptitle('Reusing one batch is most valuable in the low-sample regime')
save(fig,'same_vs_independent_partial')

# Graph alignment metrics.
m=pd.read_csv(R/'graph_metrics.csv')
metrics=m.groupby('graph').agg(explained=('explained_variance_fraction','mean'),pair=('pair_weight_fraction','mean'),cos=('exact_ng_projected_target_cosine','mean'),dim=('parameter_dimension','mean')).loc[graphs]
fig,ax=plt.subplots(figsize=(7.3,4.5));x=np.arange(4);w=.24
ax.bar(x-w,metrics.explained,width=w,label='Explained cost variance');ax.bar(x,metrics.pair,width=w,label='Retained pair weight');ax.bar(x+w,metrics.cos,width=w,label='Exact NG cosine with projected target');ax.axhline(0,linewidth=1);ax.set_xticks(x,labels=[gl[g] for g in graphs]);ax.set_ylim(-1,1.08);ax.set_ylabel('Alignment diagnostic');ax.set_title('Partial-alignment geometry at the projected target');ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8)
save(fig,'graph_alignment_metrics')

# Width3 scaling.
s=pd.read_csv(R/'width3_sample_scaling_summary.csv')
fig,ax=plt.subplots(figsize=(7.2,4.5))
for method,label in [('sampled_adam','Adam'),('sampled_star_fisher','Graph-star'),('sampled_full_fisher','Full Fisher'),('protected_ray_star','Protected target')]:
    q=s[s.method==method].sort_values('sample_budget');ax.plot(q.sample_budget,q.success_rate,marker='o',label=label)
ax.set_xscale('log',base=2);ax.set_xticks([64,256,1024],labels=['64','256','1024']);ax.set_ylim(0,1.05);ax.set_xlabel('Samples per update');ax.set_ylabel('Success fraction');ax.set_title('Width-3 finite-sample scaling');ax.grid(alpha=.25);ax.legend()
save(fig,'width3_sample_scaling')

# Storage-success Pareto for non-chain graphs.
fig,axes=plt.subplots(1,3,figsize=(12.4,4.0),sharey=True)
sel=['sampled_adam','sampled_diagonal_fisher','sampled_two_block_fisher','sampled_star_fisher','sampled_bag_fisher','sampled_full_fisher','protected_ray_star']
short={'sampled_adam':'Adam','sampled_diagonal_fisher':'Diag','sampled_two_block_fisher':'2-block','sampled_star_fisher':'Star','sampled_bag_fisher':'Bag','sampled_full_fisher':'Full','protected_ray_star':'Protected'}
for ax,g in zip(axes,['problem_tree','width2','width3']):
    q=d[(d.graph==g)&(d.method.isin(sel))]
    for _,r in q.iterrows():
        ax.scatter(r.median_stored_metric_entries,r.success_rate,s=45);ax.annotate(short[r.method],(r.median_stored_metric_entries,r.success_rate),xytext=(3,3),textcoords='offset points',fontsize=8)
    ax.set_xscale('log');ax.set_ylim(0,1.05);ax.set_title(gl[g]);ax.set_xlabel('Stored metric entries');ax.grid(alpha=.25)
axes[0].set_ylabel('Success fraction');fig.suptitle('Geometry accuracy–storage frontier at 256 samples')
save(fig,'storage_success_frontier')

# Exact natural failure despite chain representability.
chain=d[d.graph=='chain'].set_index('method');control=pd.read_csv(R/'chain_representability.csv');evalc=control[control.split=='evaluation']
fig,axes=plt.subplots(1,2,figsize=(9.6,4.0))
axes[0].bar(['Exact NG\nfrom biased starts','Sampled full\nFisher','Field-only\nsolution ray'],[chain.loc['exact_natural_oracle'].success_rate,chain.loc['sampled_full_fisher'].success_rate,1.0]);axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Fraction reaching full gap 0.1');axes[0].set_title('Chain optimization versus representability');axes[0].grid(axis='y',alpha=.25)
axes[1].scatter(evalc.beta_for_full_gap_0p1,evalc.planted_probability);axes[1].set_xlabel(r'Field-only inverse scale $\beta$');axes[1].set_ylabel(r'$p_\star$ at full gap 0.1');axes[1].set_ylim(.94,.97);axes[1].set_title('A compact chain state represents every solution');axes[1].grid(alpha=.25)
save(fig,'chain_representability_vs_optimization')

# Exact/sample first-direction alignment.
f=pd.read_csv(R/'sampled_first_step_summary.csv');fig,axes=plt.subplots(1,4,figsize=(13.5,3.9),sharey=True)
show=['sampled_adam','sampled_star_fisher','sampled_bag_fisher','sampled_full_fisher','protected_ray_star']
# bag first direction comes from separate summary; add mean values.
bag=pd.read_csv(R/'bag_diagnostic_summary.csv').set_index('graph')
for ax,g in zip(axes,graphs):
    vals=[];names=[]
    for method in show:
        if method=='sampled_bag_fisher':v=float(bag.loc[g].cos_mean)
        else:v=float(f[(f.graph==g)&(f.method==method)].mean_cosine_exact_natural.iloc[0])
        vals.append(v);names.append(short.get(method,'Protected'))
    ax.bar(range(len(vals)),vals);ax.set_xticks(range(len(vals)),labels=names,rotation=65,ha='right');ax.set_title(gl[g]);ax.set_ylim(0,1.05);ax.grid(axis='y',alpha=.25)
axes[0].set_ylabel('Cosine with exact natural direction');fig.suptitle('First-step geometric fidelity')
save(fig,'first_direction_alignment')
