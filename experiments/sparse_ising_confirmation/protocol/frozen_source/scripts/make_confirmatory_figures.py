from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault('SOURCE_DATE_EPOCH','1787616000')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
matplotlib.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
PDF={'Creator':'QBM confirmatory checkpoint','Producer':'Matplotlib','CreationDate':None,'ModDate':None}

def save(fig,name):
 fig.tight_layout();fig.savefig(F/f'{name}.png',dpi=220,bbox_inches='tight',metadata={'Software':'QBM confirmatory checkpoint'});fig.savefig(F/f'{name}.pdf',bbox_inches='tight',metadata=PDF);plt.close(fig)

agg=pd.read_csv(R/'confirmatory_aggregate.csv')
graphs=['chain','random_tree','problem_tree','full'];labels=['Native chain','Random target tree','Max-weight problem tree','Full graph']
fig,ax=plt.subplots(figsize=(8.2,4.8));x=np.arange(4);w=.24
cells=[('Adam, random',"method=='adam' and initialization=='random'"),('Adam, target biased',"method=='adam' and initialization=='target_biased'"),('Exact natural, target biased',"method=='exact_natural'")]
for j,(label,q) in enumerate(cells):
 d=agg.query(q).set_index('graph').loc[graphs];ax.bar(x+(j-1)*w,d.success_rate,w,label=label)
ax.set_xticks(x,labels,rotation=12,ha='right');ax.set_ylim(0,1.06);ax.set_ylabel('Success fraction');ax.set_title('Preregistered sparse-Ising confirmation');ax.grid(axis='y',alpha=.25);ax.legend(frameon=False)
save(fig,'confirmatory_success_by_representation')

eff=pd.read_csv(R/'confirmatory_primary_effects.csv').sort_values('id')
fig,ax=plt.subplots(figsize=(7.5,4.3));y=np.arange(len(eff));point=100*eff.point_difference;lo=100*(eff.point_difference-eff.holm_ci_low);hi=100*(eff.holm_ci_high-eff.point_difference)
ax.errorbar(point,y,xerr=np.vstack([lo,hi]),fmt='o',capsize=5);ax.axvline(0,linewidth=1);ax.set_yticks(y,[f"{r.id}: {r.label}" for r in eff.itertuples()]);ax.set_xlabel('Paired success difference (percentage points)');ax.set_title('Frozen primary comparisons with Holm-adjusted intervals');ax.grid(axis='x',alpha=.25)
save(fig,'confirmatory_primary_effects')

res=pd.read_csv(R/'confirmatory_preparation_resources.csv')
fig,ax=plt.subplots(figsize=(7.5,4.5));data=[]
for g,label in zip(graphs,labels):
 d=res[res.graph==g];data.append(d.conditional_angle_entries.to_numpy())
ax.boxplot(data,tick_labels=labels,showmeans=True);ax.set_yscale('log');ax.set_ylabel('Conditional rotation-angle entries (log scale)');ax.set_title('Exact q-sample representation cost');ax.grid(axis='y',which='both',alpha=.25);plt.setp(ax.get_xticklabels(),rotation=12,ha='right')
save(fig,'confirmatory_preparation_resources')

s=pd.read_csv(R/'confirmatory_trajectory_summary.csv')
fig,ax=plt.subplots(figsize=(7.6,4.6));x=np.arange(4);w=.32
for j,(init,label) in enumerate([('random','Random'),('target_biased','Target biased')]):
 d=s.query("method=='adam' and initialization==@init").groupby('graph').success.mean().reindex(graphs);ax.bar(x+(j-.5)*w,d,w,label=label)
ax.set_xticks(x,labels,rotation=12,ha='right');ax.set_ylim(0,.78);ax.set_ylabel('Adam success fraction');ax.set_title('Initialization effect across representations');ax.grid(axis='y',alpha=.25);ax.legend(frameon=False)
save(fig,'confirmatory_initialization_effect')

# Instance-level paired rates for H1/H2/H3
fig,axes=plt.subplots(3,1,figsize=(8.2,8.0),sharex=True)
specs=[('H1','adam','problem_tree','chain'),('H2','adam','problem_tree','random_tree'),('H3','exact_natural','problem_tree','chain')]
for ax,(hid,method,a,b) in zip(axes,specs):
 q="method==@method and initialization=='target_biased'"
 d=s.query(q);pa=d[d.graph==a].groupby('instance_id').success.mean();pb=d[d.graph==b].groupby('instance_id').success.mean();diff=pa-pb
 ax.bar(np.arange(len(diff)),diff.to_numpy());ax.axhline(0,linewidth=1);ax.set_ylabel(f'{hid} diff.');ax.grid(axis='y',alpha=.2)
axes[-1].set_xlabel('Confirmatory instance (frozen order)');fig.suptitle('Instance-level paired effects')
save(fig,'confirmatory_instance_effects')
