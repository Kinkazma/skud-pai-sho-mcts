#!/usr/bin/env python3
"""English command-line entry point; no training or package installation on import."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from scripts.prepare_models import ROOT,prepare,sha
from scripts.assets import require_groups

def run(args):subprocess.run([str(a) for a in args],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['doctor','setup','prepare','play','serve','train','pause','status','dashboard'])
 p.add_argument('--generation',default=None);p.add_argument('--opponent',default='3.1');p.add_argument('--budget',type=int,default=32);p.add_argument('--seed',type=int,default=71)
 p.add_argument('--seconds',type=int,default=60);p.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1));p.add_argument('--decisions',type=int,default=512)
 p.add_argument('--heuristic-reference',action='store_true');p.add_argument('--resume-from');p.add_argument('--output',default='runs/new');p.add_argument('--replay',action='store_true');p.add_argument('--replay-gib',type=int,default=12);p.add_argument('--port',type=int,default=8770)
 a=p.parse_args();g=a.generation or json.loads((ROOT/'release.json').read_text())['defaultGeneration']
 trainer=ROOT/'target/release/paisho-gen32';suite=ROOT/'inference/target/release/paisho-gen3-suite';output=(ROOT/a.output).resolve()
 if a.command=='doctor':
  print(json.dumps({'python':sys.version.split()[0],'cargo':shutil.which('cargo'),'platform':sys.platform,'cpus':os.cpu_count(),'generation':g,'trainer':trainer.is_file(),'inference':suite.is_file(),'assetsPresent':all((ROOT/k).exists() for k in json.loads((ROOT/'assets-manifest.json').read_text())['files'])},indent=2));return
 if a.command=='setup':
  require_groups(['core-memory'])
  if not shutil.which('cargo'):raise SystemExit('Install Rust through rustup first; see README prerequisites.')
  run(['cargo','build','--release','--locked','-p','paisho-train','--bin','paisho-gen32','--bin','paisho-compact']);run(['cargo','build','--manifest-path','inference/Cargo.toml','--release','--locked','-p','paisho-train','--bin','paisho-gen3-suite']);prepare();return
 if a.command=='dashboard':run([sys.executable,ROOT/'scripts/dashboard.py','--port',a.port]);return
 if a.command=='pause':
  if output.with_suffix('.launcher.json').exists() and json.loads(output.with_suffix('.launcher.json').read_text()).get('generation')=='3.1':raise SystemExit('The historical Gen3.1 trainer has no cooperative pause; it stops at its configured duration. Use short bounded runs.')
  if not (output/'progress.json').exists():raise SystemExit('No training progress found at this output.')
  (output/'pause-request.json').write_text('{}\n');print('Cooperative pause requested.');return
 if a.command=='status':
  for name in ['progress.json','live.json','final.json']:
   if (output/name).exists():print((output/name).read_text())
  return
 if a.command=='train' and a.replay:require_groups(['gen35-replay'])
 models=prepare()
 if a.command=='prepare':return
 if a.command=='serve':run([suite,'serve',models/'opponents.json',g,a.budget,a.seed,'gen3']);return
 if a.command=='play':
  if output.exists():raise SystemExit('Output already exists; select a new PSR file.')
  output.parent.mkdir(parents=True,exist_ok=True);run([suite,'play',models/'opponents.json',g,a.opponent,a.budget,a.budget,a.seed,0,a.decisions,output,'gen3']);return
 if a.command=='train':
  if g=='3.1':
   if a.replay or a.resume_from:raise SystemExit('Gen3.1 replay continuation uses paisho-compact selfplay --replay-input; see docs/TRAINING.md.')
   if output.exists():raise SystemExit('Output already exists')
   output.parent.mkdir(parents=True,exist_ok=True);output.with_suffix('.launcher.json').write_text(json.dumps({'generation':'3.1','seconds':a.seconds})+'\n')
   run([ROOT/'target/release/paisho-compact','selfplay','--model',models/'gen3-1.json','--output',output,'--seconds',a.seconds,'--workers',a.workers,'--simulations',a.budget,'--decision-limit',a.decisions,'--seed',a.seed]);return
  if g not in ['3.2','3.3','3.4','3.5']:raise SystemExit('Unsupported generation')
  if output.exists() or output.with_suffix('.config.json').exists():raise SystemExit('Use a new output directory; never overwrite a previous run.')
  if a.seconds<1 or a.workers<1 or a.replay_gib<1:raise SystemExit('Positive resource settings required')
  config=json.loads((ROOT/'configs/gen3.5-historical.json').read_text())
  config.update(model=str(models/f"gen{g.replace('.','-')}.json"),output=str(output),seconds=a.seconds,threads=a.workers,actors=a.workers,archive_workers=1,budgets=[a.budget],caps=[5.0],decisions=a.decisions,seed=a.seed,checkpoint_seconds=10,replay_max_bytes=a.replay_gib*1024**3,replay_index=str(models/'replay-gen3.5.json') if a.replay else None)
  if a.replay and g!='3.5':raise SystemExit('The shipped full replay belongs to Gen3.5; do not attach it silently to another lineage.')
  if a.heuristic_reference:config['historical_pool']=[]
  for ref in config['historical_pool']:
   ref['path']=str(models/'references'/Path(ref['path']).name);ref['sha256']=sha(Path(ref['path']))
  if a.resume_from:
   prior=(ROOT/a.resume_from).resolve();previous=json.loads((prior/'checkpoint.json').read_text())
   config['model']=previous['model'];config['replay_index']=previous['replay_index']
  output.parent.mkdir(parents=True,exist_ok=True);cfg=output.with_suffix('.config.json');cfg.write_text(json.dumps(config,indent=2)+'\n');run([trainer,'run',cfg])
if __name__=='__main__':main()
