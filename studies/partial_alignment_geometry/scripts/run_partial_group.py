from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

STUDY=Path(__file__).resolve().parents[1];REPOSITORY=Path(__file__).resolve().parents[3];RESULTS=REPOSITORY/'results'/'partial_alignment_geometry';INSTANCES=REPOSITORY/'data'/'certificate_tight_instances';GRAPHS=STUDY/'graphs'
sys.path.insert(0,str(STUDY/'scripts'));sys.path.insert(0,str(REPOSITORY/'src'))
from partial_alignment_study import generate_family, worker, MAIN_METHODS, SCALING_METHODS

instance_id=sys.argv[1]; graph=sys.argv[2]; budget=int(sys.argv[3]); mode=sys.argv[4]
instances={x.instance_id:x for x in generate_family(INSTANCES)}
graphs=json.loads((GRAPHS/'partial_graphs.json').read_text())
methods=MAIN_METHODS if mode=='main' else SCALING_METHODS
rows,logs=worker((instances[instance_id],graph,budget,methods,graphs))
stem=f'{instance_id}__{graph}__M{budget}__{mode}'
out=RESULTS/'groups';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/f'{stem}_trajectories.csv',index=False)
pd.DataFrame(logs).to_csv(out/f'{stem}_logs.csv.gz',index=False,compression='gzip')
print(stem, len(rows), len(logs), flush=True)
