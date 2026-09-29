#!/usr/bin/env python3
"""Verify local release assets and create idempotent machine-local model paths."""
import hashlib,json,sys
from pathlib import Path
if __package__:
 from .assets import require_groups
else:
 from assets import require_groups
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def relocate(value):
 if isinstance(value,list):return [relocate(v) for v in value]
 if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
 if isinstance(value,str) and value.startswith(('memory/','assets/','models/','data/')) and (ROOT/value).is_file():return str((ROOT/value).resolve())
 return value

def prepare():
 require_groups(['core-memory'])
 manifest=json.loads((ROOT/'assets-manifest.json').read_text())
 for name,entry in manifest['files'].items():
  path=ROOT/name
  if not path.is_file():raise FileNotFoundError(f'Missing release asset: {name}. Install the matching data pack.')
  if sha(path)!=entry['sha256']:raise ValueError('Asset checksum mismatch: '+name)
 out=ROOT/'portable-models';out.mkdir(exist_ok=True);opponents=[]
 for path in sorted((ROOT/'models').rglob('*.json')):
  model=relocate(json.loads(path.read_text()));memory=model.get('memory_manifest')
  if memory:model['memory_manifest_sha256']=sha(Path(memory))
  dest=out/path.relative_to(ROOT/'models');dest.parent.mkdir(parents=True,exist_ok=True)
  dest.write_text(json.dumps(model,separators=(',',':'))+'\n')
  if path.parent.name=='models':
   generation=path.stem.removeprefix('gen').replace('-','.')
   opponents.append({'generation':generation,'model':str(dest),'sha256':sha(dest),'solver':generation not in ['3.1','3.2']})
 (out/'opponents.json').write_text(json.dumps(opponents,indent=2)+'\n')
 index=ROOT/'assets/gen3.5-replay/replay.json'
 if index.exists():
  data=relocate(json.loads(index.read_text()));(out/'replay-gen3.5.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
 print(f'Prepared {len(opponents)} models; numerical weights unchanged.',file=sys.stderr)
 return out
if __name__=='__main__':prepare()
