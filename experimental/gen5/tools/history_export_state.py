#!/usr/bin/env python3
"""Prepare neutral Gen5 checkpoint dependencies without running training."""
import argparse, gzip, hashlib, json, re, sqlite3
from pathlib import Path
from export_history import atomic, sha, sanitize, export_one

STR = re.compile(rb'"(?:[^"\\]|\\.)*"')

def prepare(config_path, root, journal):
    config=json.loads(config_path.read_text());progress_path=Path(config['resume_progress'])
    progress=json.loads(progress_path.read_text());base=root/'assets/gen5-history'
    state=base/'state';state.mkdir(parents=True,exist_ok=True)
    mapping={};hashes={};copied=[]
    def rel(p):return str(p.relative_to(root))
    def add(old,dest):mapping[str(old)]=rel(dest)
    def neutral(data):
        def replace(m):
            s=json.loads(m[0])
            if s in mapping:return json.dumps(mapping[s],ensure_ascii=False).encode()
            if '/Users/' in s:return json.dumps('source-artifact:'+sha(s.encode())).encode()
            return m[0]
        return STR.sub(replace,data)
    learner=Path(config['model']);actor=Path(progress['publication_guard']['accepted_path'])
    a=json.loads(learner.read_text());add(a['sequence_memory']['path'],root/'memory/gen5-sequences-v23/memory.bin')
    for opponent in config['opponents']:
        model=json.loads(Path(opponent['model']).read_text())
        if model.get('memory_manifest'):
            original=Path(model['memory_manifest'])
            add(original,root/'memory'/original.parent.name/original.name)
            bank=model['policy']['sequence_memory']
            destination=root/'memory'/original.parent.name/'memory.bin'
            if sha(destination.read_bytes()) != bank['sha256']:
                raise ValueError('Frozen opponent memory bank differs')
            add(bank['path'],destination)
    # Reuse the already exported human corpus with its exact record hashes.
    add(config['human_dataset'],root/'experimental/gen5/inputs/human-dataset.json')
    sources={learner:state/'learner.json',actor:state/'accepted.json'}
    add(progress['checkpoint_model'],state/'learner.json')
    for key in ['reference','evaluation_anchor','publication_guard','publication_validation']:
        sources[Path(config[key])]=state/'inputs'/Path(config[key]).name
    for o in config['opponents']:sources[Path(o['model'])]=state/'inputs'/Path(o['model']).name
    for i,e in enumerate(progress['frozen_evaluations']):sources[Path(e['path'])]=state/'frozen'/f'reference-{i}.json'
    for checkpoint in [progress['protection']['checkpoint'],progress['durable_recall']['deferred_checkpoint'],progress['durable_recall']['structured']['checkpoint']]:
        src=Path(checkpoint['path'])
        if sha(src.read_bytes()) != checkpoint['sha256']:raise ValueError('Checkpoint hash differs')
        sources[src]=state/'checkpoints'/src.name
    def pending(v):
        if isinstance(v,dict):
            if v.get('pending_record'):
                path,expected=v['pending_record'];src=Path(path)
                if sha(src.read_bytes())!=expected:raise ValueError('Pending record hash mismatch')
                sources[src]=state/'pending'/src.name
            for x in v.values():pending(x)
        elif isinstance(v,list):
            for x in v:pending(x)
    pending(progress['curriculum_states'])
    for src,dest in sources.items():add(src,dest)
    for src,dest in sources.items():
        raw=src.read_bytes();out=neutral(raw)
        atomic(dest,out);hashes[sha(raw)]=sha(out);copied.append({'path':rel(dest),'sha256':sha(out),'source_sha256':sha(raw),'bytes':len(out)})
    # Both prepared repairs and the unmodified last campaign protocol are recorded.
    anchors=Path(config['structured_recall_anchors']['path']);dest=state/'structured-anchors.json'
    atomic(dest,neutral(anchors.read_bytes()));add(anchors,dest)
    config['structured_recall_anchors']['sha256']=sha(dest.read_bytes())
    index_path=Path(config['replay_index']);index=json.loads(index_path.read_text())
    fifo=[]
    for path,expected in sorted({r['source']['path']:r['source']['sha256'] for r in index['rows']}.items()):
        dest=base/'fifo'/(expected+'.json.gz');fifo.append((Path(path),dest,'fifo',expected))
    atomic(state/'fifo-jobs.private.json',json.dumps([(str(x),str(y),k,h) for x,y,k,h in fifo]).encode())
    # Private file is removed after job completion and never listed for release.
    add(config['replay_index'],state/'replay.index.json');add(progress['checkpoint_replay_index'],state/'replay.index.json')
    add(progress_path,state/'resume-progress.json')
    for lane in ['proofs','ordinary']:
        add(progress['durable_recall']['coverage'][lane]['cursor']['manifest'],state/f'coverage-{lane}.json')
    old_config=config.copy()
    config=json.loads(neutral(json.dumps(config,separators=(',',':')).encode()))
    config['case_curriculum']['archive']=rel(base/'archives'/Path(old_config['case_curriculum']['archive']).name)
    config['recall_archive_sources']=[rel(base/'archives'/Path(p).name) for p in old_config['recall_archive_sources']]
    config['seconds']=progress['remaining_seconds'];config['end_unix_seconds']=None
    config['output']='runs/gen5-historical-continuation'
    progress=json.loads(neutral(json.dumps(progress,separators=(',',':')).encode()))
    atomic(state/'config.prepared.json',(json.dumps(config,indent=2)+'\n').encode())
    atomic(state/'progress.partial.json',json.dumps(progress,separators=(',',':')).encode())
    # Native finalization recalculates typed model/registry identities; no weight changes.
    old_cases=Path(old_config['output'])/'human-cases.json'
    atomic(state/'human-cases.original.json',neutral(old_cases.read_bytes()))
    original=json.loads(progress_path.read_text())
    atomic(state/'export-inputs.json',json.dumps({'files':copied,'source_progress_sha256':sha(progress_path.read_bytes()),'source_config_sha256':sha(config_path.read_bytes()),'source_actor_identity':original['publication_guard']['accepted_identity'],'source_case_manifest':original['case_manifest_sha256'],'source_learner_parameters_sha256':sha(re.search(rb'"parameters"\s*:\s*(\[[^\]]*\])',learner.read_bytes())[1])},indent=2).encode())
    private={'config':str(config_path),'root':str(root),'path_map':mapping,'hash_map':hashes,'progress':str(progress_path)}
    atomic(journal.with_suffix('.state.json'),json.dumps(private).encode())
    print(json.dumps({'small_files':len(copied),'fifo_jobs':len(fifo),'state':str(state)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('root',type=Path);p.add_argument('--journal',type=Path,required=True)
    a=p.parse_args();prepare(a.config,a.root,a.journal)
