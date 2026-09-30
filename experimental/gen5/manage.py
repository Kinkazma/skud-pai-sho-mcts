#!/usr/bin/env python3
"""Gen5 research commands. For full historical state use tools/continue_history.py."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.assets import require_groups
EXP=ROOT/'experimental/gen5'
BINARY=EXP/'target/release/paisho-gen5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def localize(v):
 if isinstance(v,list):return [localize(x) for x in v]
 if isinstance(v,dict):return {k:localize(x) for k,x in v.items()}
 if isinstance(v,str) and v.startswith(('memory/','assets/','experimental/','data/')) and (ROOT/v).is_file():return str(ROOT/v)
 return v

def prepare(training=True):
 require_groups(['core-memory','gen5-memory'] + (['human','gen5-learning'] if training else []))
 out=ROOT/'portable-models/gen5';out.mkdir(parents=True,exist_ok=True)
 for folder in ['models','inputs']:
  for p in (EXP/folder).glob('*.json'):
   if not training and p.name=='human-dataset.json':continue
   value=localize(json.loads(p.read_text()))
   if value.get('memory_manifest'):value['memory_manifest_sha256']=sha(Path(value['memory_manifest']))
   d=out/folder/p.name;d.parent.mkdir(exist_ok=True);d.write_text(json.dumps(value,separators=(',',':'))+'\n')
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['build','prepare','compare','train']);p.add_argument('--seconds',type=float,default=30);p.add_argument('--workers',type=int,default=2);p.add_argument('--output',default='runs/gen5-new');p.add_argument('--budget',type=int,default=256);p.add_argument('--decisions',type=int,default=800)
 a=p.parse_args()
 if a.command=='build':subprocess.run(['cargo','build','--manifest-path',str(EXP/'Cargo.toml'),'--release','--locked','-p','paisho-train','--bin','paisho-gen5'],cwd=ROOT,check=True);return
 if not 0<a.seconds<=86400 or not 1<=a.workers<=64 or not 1<=a.decisions<=800:raise SystemExit('Positive bounded duration, workers and decisions required')
 local=prepare(training=a.command!='compare')
 if a.command=='prepare':return
 output=(ROOT/a.output).resolve();cfg=output.with_suffix('.config.json')
 if output.exists() or cfg.exists():raise SystemExit('Select a new output; this command never resumes or overwrites an existing campaign.')
 if a.command=='compare':
  c={'model':str(local/'models/accepted.json'),'reference':str(local/'inputs/reference-gen3-1.json'),'output':str(output),'seconds':a.seconds,'threads':a.workers,'workers':a.workers,'budget':a.budget,'reference_budget':a.budget,'pairs':1,'decisions':a.decisions,'game_seconds':a.seconds,'seed':97500}
 else:
  if a.budget not in [256,512]:raise SystemExit('Gen5 training budgets: 256 or 512')
  c=json.loads((EXP/'configs/historical-template.json').read_text());c=localize(c)
  for k in ['reference','human_dataset','evaluation_anchor','publication_guard','publication_validation']:c[k]=str(local/'inputs'/Path(c[k]).name)
  for o in c['opponents']:
   o['model']=str(local/'inputs'/Path(o['model']).name);o['sha256']=sha(Path(o['model']))
  c.update(model=str(local/'models/accepted.json'),output=str(output),seconds=a.seconds,end_unix_seconds=None,threads=a.workers,actors=a.workers,secondary_threads=1,search_pool_shards=1,games=10000000,budgets=[[a.budget,1.0]],decision_limit=a.decisions,replay_index=None,resume_progress=None,recall_archive_sources=[str(ROOT/'assets/gen5-seed-cases')],checkpoint_seconds=10,replay_max_bytes=2*1024**3,replay_capacity=8192)
  c['case_curriculum']['archive']=str(output/'durable')
  anchors=ROOT/'assets/gen5-replay/structured-anchors.json';c['structured_recall_anchors']={'path':str(anchors),'sha256':sha(anchors)}
  print('New experiment from accepted weights + guard panels and 128 structured anchors; historical FIFO, durable archive and campaign schedule are NOT resumed.',file=sys.stderr)
 output.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(json.dumps(c,indent=2)+'\n')
 subprocess.run([str(BINARY),'compare' if a.command=='compare' else 'run',str(cfg)],cwd=ROOT,check=True)
if __name__=='__main__':main()
