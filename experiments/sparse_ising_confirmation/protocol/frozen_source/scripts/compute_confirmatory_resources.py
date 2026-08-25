from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from confirmatory_common import build_problem,load_manifest

def main():
    instances=load_manifest(ROOT/'instances'/'confirmatory_sparse_ising_manifest.json')
    rows=[]
    for instance in instances:
        problem=build_problem(instance,compute_resources=True)
        for graph,representation in problem.representations.items():
            rows.append({'instance_id':instance.instance_id,'graph':graph,'edge_count':len(representation.edges),'parameter_count':representation.features.shape[1],**representation.resources})
        print('resources',instance.instance_id,flush=True)
    pd.DataFrame(rows).sort_values(['instance_id','graph']).to_csv(ROOT/'results'/'confirmatory_preparation_resources.csv',index=False)
if __name__=='__main__':main()
