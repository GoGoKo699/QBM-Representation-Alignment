import sys
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / 'src'))
STUDY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / 'results' / 'boundary_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'
OUT = RESULTS
import pandas as pd, numpy as np
from qbm_alignment.certificate_family import generate_family
from qbm_alignment.optimizer_geometry import build,Adam,evaluate,geom
TARGETS={'ct_w5_i1','ct_w6_i1'}
rows=[]
for inst in [x for x in generate_family(INSTANCES) if x.instance_id in TARGETS]:
    P=build(inst);theta=P.c.copy();adam=Adam(len(theta));E,g,p,_=evaluate(theta,P.F,P.C,False);g0=float(np.sqrt(np.mean(g*g)))
    for step in range(1000):
        E,g,p,_=evaluate(theta,P.F,P.C,False);dom=int(np.argmax(p));beta=float(theta@P.c/(P.c@P.c));u=theta-beta*P.c;H=float(-np.sum(p[p>0]*np.log(p[p>0])));upd=adam.step(g) if step<999 else np.zeros_like(theta)
        rows.append({'instance_id':inst.instance_id,'step':step,'gap':E-P.ground,'pstar':float(p[P.pidx]),'dominant_index':dom,'dominant_probability':float(p[dom]),'dominant_gap':float(P.C[dom]-P.ground),'entropy':H,'effective_support':float(np.exp(H)),'gradient_rms':float(np.sqrt(np.mean(g*g))),'gradient_ratio':float(np.sqrt(np.mean(g*g))/g0),'theta_target_cosine':float(theta@P.c/(np.linalg.norm(theta)*np.linalg.norm(P.c))),'beta_projection':beta,'transverse_norm':float(np.linalg.norm(u)),'transverse_ratio':float(np.linalg.norm(u)/max(abs(beta)*np.linalg.norm(P.c),1e-300)),'update_target_cosine':float(upd@P.c/(np.linalg.norm(upd)*np.linalg.norm(P.c))) if np.linalg.norm(upd)>0 else np.nan})
        if step<999:theta+=upd
    print('dense',inst.instance_id,flush=True)
pd.DataFrame(rows).to_csv(OUT/'baseline_dense_trajectory.csv.gz',index=False,compression='gzip')
