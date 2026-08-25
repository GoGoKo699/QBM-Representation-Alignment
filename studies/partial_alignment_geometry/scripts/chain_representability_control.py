from __future__ import annotations
import json,sys,math
from pathlib import Path
import numpy as np,pandas as pd
STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs';sys.path.insert(0,str(REPOSITORY/'src'));sys.path.insert(0,str(STUDY/'scripts'))
from partial_alignment_study import generate_family,build_problem
from qbm_alignment.certificate_family import evaluate

def beta_for_gap(problem,target=.1):
    zstar=1-2*np.asarray(problem.planted,dtype=float);direction=np.r_[-zstar,np.zeros(len(problem.edges))]
    def vals(beta):
        E,g,p,I=evaluate(beta*direction,problem.F,problem.C,False);return float(E-problem.ground),float(p[problem.pidx])
    lo=0.;hi=1.
    while vals(hi)[0]>target:hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if vals(mid)[0]<=target:hi=mid
        else:lo=mid
    gap,p=vals(hi);return hi,gap,p,direction

def main():
    fam=generate_family(INSTANCES);graphs=json.loads((GRAPHS/'partial_graphs.json').read_text());rows=[]
    for inst in fam:
        p=build_problem(inst,'chain',graphs);beta,gap,pstar,d=beta_for_gap(p)
        analytic=(1/(1+math.exp(-2*beta)))**p.n
        rows.append({'instance_id':p.instance_id,'instance_width':p.instance_width,'split':p.split,'n':p.n,'chain_parameter_dimension':len(d),'beta_for_full_gap_0p1':beta,'full_gap_check':gap,'planted_probability':pstar,'analytic_product_probability':analytic,'probability_residual':abs(pstar-analytic)})
    d=pd.DataFrame(rows);d.to_csv(RESULTS/'chain_representability_control.csv',index=False)
    print(d.groupby(['split','instance_width']).agg(instances=('instance_id','size'),beta_min=('beta_for_full_gap_0p1','min'),beta_mean=('beta_for_full_gap_0p1','mean'),beta_max=('beta_for_full_gap_0p1','max'),pstar_min=('planted_probability','min'),pstar_mean=('planted_probability','mean')).to_string())
if __name__=='__main__':main()
