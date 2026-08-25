import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / 'src'))
STUDY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / 'results' / 'boundary_geometry'
INSTANCES = REPOSITORY / 'data' / 'certificate_tight_instances'
OUT = RESULTS
import pickle
from qbm_alignment.certificate_family import generate_family
from qbm_alignment.optimizer_geometry import run_baselines
f=generate_family(INSTANCES)
s,l,states=run_baselines(f)
s.to_csv(OUT/'baseline_summary.csv',index=False);l.to_csv(OUT/'baseline_geometry.csv',index=False)
with (OUT/'baseline_states.pkl').open('wb') as h:pickle.dump(states,h)
print(s.to_string(index=False))
print(l[l.trap].to_string(index=False))
