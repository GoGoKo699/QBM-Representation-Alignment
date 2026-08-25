from __future__ import annotations
from pathlib import Path
import os
os.environ.setdefault('SOURCE_DATE_EPOCH','1787529600')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];R=REPOSITORY/'results'/'finite_sample_geometry';F=STUDY/'figures';F.mkdir(exist_ok=True)
PDF_META={'Creator':'QBM finite-sample geometry study','Producer':'Matplotlib','CreationDate':None,'ModDate':None};PNG_META={'Software':'QBM finite-sample geometry study'}
LABELS={'sampled_adam':'Sampled Adam','sampled_diagonal_fisher':'Diagonal Fisher','sampled_two_block_fisher':'Two-block Fisher','sampled_full_fisher':'Full sample Fisher','ray_plus_residual':'Ray + residual','analytic_target_cooling':'Analytic target cooling'}

def save(fig,name):
 fig.tight_layout();fig.savefig(F/f'{name}.pdf',bbox_inches='tight',metadata=PDF_META);fig.savefig(F/f'{name}.png',dpi=220,bbox_inches='tight',metadata=PNG_META);plt.close(fig)

agg=pd.read_csv(R/'broad_aggregate_evaluation.csv');ci=pd.read_csv(R/'evaluation_cluster_bootstrap_rates.csv');direction=pd.read_csv(R/'direction_rank_storage_summary.csv');trap=pd.read_csv(R/'trap_replay_aggregate.csv');fullrank=pd.read_csv(R/'full_fisher_rank_success.csv');ind=pd.read_csv(R/'independent_full_fisher_aggregate.csv');ind_eval=ind[ind.split=='evaluation']
methods=['sampled_adam','sampled_diagonal_fisher','sampled_two_block_fisher','sampled_full_fisher','ray_plus_residual'];budgets=[64,256,1024,4096]

# Success with cluster intervals.
fig,ax=plt.subplots(figsize=(7.4,4.7))
for method in methods:
 g=ci[ci.method==method].sort_values('sample_budget');y=g.success_rate.to_numpy();lo=y-g.ci_low.to_numpy();hi=g.ci_high.to_numpy()-y
 ax.errorbar(g.sample_budget,y,yerr=np.vstack([lo,hi]),marker='o',capsize=3,label=LABELS[method])
ax.plot(ind_eval.sample_budget,ind_eval.success_rate,marker='x',linestyle='--',label='Independent-batch full Fisher')
ax.axhline(1.0,linestyle=':',linewidth=1,label='Analytic target cooling')
ax.set_xscale('log',base=2);ax.set_xticks(budgets,labels=[str(x) for x in budgets]);ax.set_ylim(-.03,1.05);ax.set_xlabel('Gibbs samples per update');ax.set_ylabel('Success fraction');ax.set_title('Finite-sample success on 16 held-out instances');ax.grid(True,alpha=.25);ax.legend(frameon=False,ncol=2)
save(fig,'evaluation_success_vs_samples')

# Samples consumed per observed success.
fig,ax=plt.subplots(figsize=(7.3,4.7))
for method in methods:
 g=agg[agg.method==method].sort_values('sample_budget');ax.plot(g.sample_budget,g.samples_per_observed_success,marker='o',label=LABELS[method])
ax.plot(ind_eval.sample_budget,ind_eval.samples_per_observed_success,marker='x',linestyle='--',label='Independent-batch full Fisher')
ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(budgets,labels=[str(x) for x in budgets]);ax.set_xlabel('Gibbs samples per update');ax.set_ylabel('Total consumed samples / observed success');ax.set_title('Empirical sample-efficiency frontier');ax.grid(True,which='both',alpha=.25);ax.legend(frameon=False,ncol=2)
save(fig,'samples_per_observed_success')

# First direction cosine.
fig,ax=plt.subplots(figsize=(7.3,4.7))
for method in methods:
 g=direction[direction.method==method].sort_values('sample_budget');ax.plot(g.sample_budget,g.mean_direction_cosine,marker='o',label=LABELS[method])
ax.set_xscale('log',base=2);ax.set_xticks(budgets,labels=[str(x) for x in budgets]);ax.set_ylim(-.02,1.03);ax.set_xlabel('Gibbs samples per update');ax.set_ylabel(r'Mean cosine with target direction $c$');ax.set_title('Estimated update alignment');ax.grid(True,alpha=.25);ax.legend(frameon=False,ncol=2)
save(fig,'direction_alignment_vs_samples')

# Full Fisher rank and success.
fig,ax1=plt.subplots(figsize=(7.1,4.6));g=direction[direction.method=='sampled_full_fisher'].sort_values('sample_budget')
ax1.plot(g.sample_budget,g.mean_rank_deficit,marker='o',label='Mean rank deficit');ax1.set_xscale('log',base=2);ax1.set_yscale('log');ax1.set_xticks(budgets,labels=[str(x) for x in budgets]);ax1.set_xlabel('Gibbs samples per update');ax1.set_ylabel('Mean covariance rank deficit')
ax2=ax1.twinx();ax2.plot(g.sample_budget,g.success_rate,marker='s',linestyle='--',label='Success fraction');ax2.set_ylim(.85,1.01);ax2.set_ylabel('Success fraction')
lines=ax1.lines+ax2.lines;ax1.legend(lines,[x.get_label() for x in lines],frameon=False,loc='center right');ax1.set_title('Full Fisher: rank recovery controls finite-sample performance');ax1.grid(True,which='both',alpha=.25)
save(fig,'full_fisher_rank_recovery')

# Storage-success Pareto at 256 and 1024.
fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharey=True)
for ax,M in zip(axes,[256,1024],strict=True):
 g=agg[agg.sample_budget==M]
 for _,r in g.iterrows():
  ax.scatter(r.mean_storage_entries,r.success_rate,s=55);ax.annotate(LABELS[r.method],(r.mean_storage_entries,r.success_rate),xytext=(4,3),textcoords='offset points',fontsize=8)
 ax.set_xscale('log');ax.set_ylim(.35,1.03);ax.set_xlabel('Stored optimizer/Fisher entries (log scale)');ax.set_title(f'{M} samples/update');ax.grid(True,alpha=.25)
axes[0].set_ylabel('Success fraction');fig.suptitle('Accuracy–memory tradeoff on held-out instances')
save(fig,'storage_success_pareto')

# Trap replay success.
fig,axes=plt.subplots(1,2,figsize=(11.4,4.4),sharey=True)
for ax,checkpoint in zip(axes,[199,999],strict=True):
 for method in ['sampled_adam','sampled_full_fisher','ray_plus_residual']:
  g=trap[(trap.checkpoint==checkpoint)&(trap.method==method)].sort_values('sample_budget');ax.plot(g.sample_budget,g.success_rate,marker='o',label=LABELS[method])
 ax.axhline(1.0,linestyle=':',linewidth=1,label='Analytic target cooling');ax.set_xscale('log',base=2);ax.set_xticks(budgets,labels=[str(x) for x in budgets]);ax.set_ylim(-.03,1.03);ax.set_xlabel('Gibbs samples per update');ax.set_title(f'Replay from record {checkpoint}');ax.grid(True,alpha=.25)
axes[0].set_ylabel('Rescue fraction');axes[1].legend(frameon=False,loc='center left');fig.suptitle('Finite-sample replay of excited-boundary traps')
save(fig,'trap_replay_success')

# Trap sample rank.
fig,ax=plt.subplots(figsize=(7.1,4.5))
for checkpoint in [199,999]:
 g=trap[(trap.checkpoint==checkpoint)&(trap.method=='sampled_full_fisher')].sort_values('sample_budget');ax.plot(g.sample_budget,g.mean_initial_rank,marker='o',label=f'Record {checkpoint}')
ax.set_xscale('log',base=2);ax.set_xticks(budgets,labels=[str(x) for x in budgets]);ax.set_xlabel('Gibbs samples per update');ax.set_ylabel('Mean sampled covariance rank');ax.set_title('Boundary concentration prevents Fisher-rank recovery');ax.grid(True,alpha=.25);ax.legend(frameon=False)
save(fig,'trap_sample_rank')

# Trap rates in broad study.
fig,ax=plt.subplots(figsize=(7.2,4.5))
for method in ['sampled_adam','sampled_diagonal_fisher','sampled_two_block_fisher','sampled_full_fisher','ray_plus_residual']:
 g=agg[agg.method==method].sort_values('sample_budget');ax.plot(g.sample_budget,g.trap_rate,marker='o',label=LABELS[method])
ax.set_xscale('log',base=2);ax.set_xticks(budgets,labels=[str(x) for x in budgets]);ax.set_ylim(-.02,.62);ax.set_xlabel('Gibbs samples per update');ax.set_ylabel('Final strong-trap fraction');ax.set_title('More samples do not repair coordinatewise methods');ax.grid(True,alpha=.25);ax.legend(frameon=False,ncol=2)
save(fig,'broad_trap_rate')

# Coupled versus independent full-Fisher estimators.
fig,axes=plt.subplots(1,2,figsize=(11.3,4.5))
same=agg[agg.method=='sampled_full_fisher'].sort_values('sample_budget')
axes[0].plot(same.sample_budget,same.success_rate,marker='o',label='Same batch')
axes[0].plot(ind_eval.sample_budget,ind_eval.success_rate,marker='s',label='Independent batches')
axes[0].set_xscale('log',base=2);axes[0].set_xticks(budgets,labels=[str(x) for x in budgets]);axes[0].set_ylim(.1,1.03);axes[0].set_xlabel('Total Gibbs samples per update');axes[0].set_ylabel('Success fraction');axes[0].set_title('Success');axes[0].grid(True,alpha=.25);axes[0].legend(frameon=False)
sg=direction[direction.method=='sampled_full_fisher'].sort_values('sample_budget')
axes[1].plot(sg.sample_budget,sg.mean_direction_cosine,marker='o',label='Same batch')
axes[1].plot(ind_eval.sample_budget,ind_eval.mean_direction_cosine,marker='s',label='Independent batches')
axes[1].set_xscale('log',base=2);axes[1].set_xticks(budgets,labels=[str(x) for x in budgets]);axes[1].set_ylim(.1,1.03);axes[1].set_xlabel('Total Gibbs samples per update');axes[1].set_ylabel(r'Mean cosine with target $c$');axes[1].set_title('Direction recovery');axes[1].grid(True,alpha=.25);axes[1].legend(frameon=False)
fig.suptitle('Using one batch for both moments cancels sampling noise algebraically')
save(fig,'same_vs_independent_full_fisher')
