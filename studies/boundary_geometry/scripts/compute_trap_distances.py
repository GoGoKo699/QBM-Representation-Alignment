import sys
import math,pickle
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / 'src'))
STUDY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / 'results' / 'boundary_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'
import pandas as pd
from qbm_alignment.certificate_family import generate_family
from qbm_alignment.optimizer_geometry import build, evaluate, TOL

def distance(P,theta):
    def gap(delta):return evaluate(theta+delta*P.c,P.F,P.C,False)[0]-P.ground
    if gap(0)<=TOL:return 0.
    lo,hi=0.,1.
    while gap(hi)>TOL:
        hi*=2
        if hi>4096:raise RuntimeError
    for _ in range(80):
        m=(lo+hi)/2
        if gap(m)<=TOL:hi=m
        else:lo=m
    return hi

def main():
    fam={x.instance_id:x for x in generate_family(INSTANCES)}
    with (OUT/'baseline_states.pkl').open('rb') as f:states=pickle.load(f)
    rows=[]
    for iid,ss in states.items():
        P=build(fam[iid])
        for cp in (199,999):
            theta=ss[cp];E,g,p,_=evaluate(theta,P.F,P.C,False);dom=int(p.argmax());beta=float(theta@P.c/(P.c@P.c));proj=beta*P.c;Ep,gp,pp,_=evaluate(proj,P.F,P.C,False)
            rows.append({'instance_id':iid,'checkpoint':cp,'starting_gap':E-P.ground,'starting_pstar':p[P.pidx],'dominant_probability':p[dom],'dominant_to_planted_log_odds':math.log(p[dom]/p[P.pidx]),'minimum_target_increment_to_success':distance(P,theta),'fixed_1000_state_increment':20.0,'projection_displacement':float(((theta-proj)**2).sum()**.5),'projected_beta':beta,'projected_gap':Ep-P.ground,'projected_pstar':pp[P.pidx],'minimum_increment_after_projection':distance(P,proj)})
    pd.DataFrame(rows).to_csv(OUT/'trap_target_distance.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
