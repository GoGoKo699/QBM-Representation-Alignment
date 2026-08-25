#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
payload=json.loads((ROOT/'CONFIRMATORY_SEED_COMMITMENT.json').read_text())
compact={
    'master_seed':payload['master_seed'],
    'engineering_instance_seeds':payload['engineering_instance_seeds'],
    'confirmatory_instance_seeds':payload['confirmatory_instance_seeds'],
}
raw=json.dumps(compact,sort_keys=True,separators=(',',':')).encode()
digest=hashlib.sha256(raw).hexdigest()
assert digest==payload['canonical_json_sha256'],(digest,payload['canonical_json_sha256'])
assert len(set(payload['engineering_instance_seeds']+payload['confirmatory_instance_seeds']))==28
assert len(payload['engineering_instance_seeds'])==4
assert len(payload['confirmatory_instance_seeds'])==24
print('Seed commitment validation passed.')
print(digest)
