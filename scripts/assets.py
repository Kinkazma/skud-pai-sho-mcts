#!/usr/bin/env python3
"""Verify or package local data packs. Never downloads or uploads anything."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','pack']);p.add_argument('group');p.add_argument('--output',default='packs');a=p.parse_args()
 manifest=ROOT/'data/manifests'/(a.group+'.json')
 if not manifest.is_file():raise SystemExit('Unknown data group; see data/ASSETS.md')
 entries=json.loads(manifest.read_text())['files']
 for rel,e in entries.items():
  f=ROOT/rel
  if not f.is_file() or f.stat().st_size!=e['bytes'] or digest(f)!=e['sha256']:raise SystemExit('Missing or altered asset: '+rel)
 if a.command=='verify':print(f'Verified {len(entries)} files in {a.group}');return
 out=(ROOT/a.output).resolve();out.mkdir(parents=True,exist_ok=True);parts=[];batch=[];size=0
 for rel,e in entries.items():
  if batch and size+e['bytes']>1024**3:parts.append(batch);batch=[];size=0
  batch.append(rel);size+=e['bytes']
 if batch:parts.append(batch)
 paths=[out/f'{a.group}-{i+1:02d}.tar' for i in range(len(parts))]
 if any(x.exists() for x in paths) or (out/(a.group+'.json')).exists():raise SystemExit('Pack output exists; choose a new output folder')
 def metadata(info):info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;return info
 result=[]
 for path,files in zip(paths,parts):
  with tarfile.open(path,'w') as archive:
   for rel in files:archive.add(ROOT/rel,arcname=rel,recursive=False,filter=metadata)
  result.append({'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path)})
 (out/(a.group+'.json')).write_text(json.dumps({'group':a.group,'parts':result},indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
