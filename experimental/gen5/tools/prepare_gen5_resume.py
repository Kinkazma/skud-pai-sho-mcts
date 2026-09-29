#!/usr/bin/env python3
"""Recover the retained FIFO from immutable receipts after the native PID is paused.

This does not signal or launch anything. The caller must suspend the verified
native process before calling prepare; files must remain unchanged throughout.
"""
import hashlib
import json
from collections import deque
from pathlib import Path
if __package__:
    from .paisho_gen5_seats import candidate_seat
else:
    from paisho_gen5_seats import candidate_seat


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def native_checkpoint(progress):
    """Read the authoritative checkpoint, never reconstruct its FIFO from receipts.

    Older writers did not publish an index pointer. Only that absence permits
    the legacy reconstruction; a missing or invalid named checkpoint is an error.
    """
    if 'checkpoint_replay_index' not in progress:return None
    if not progress.get('checkpoint_replay_index') or not progress.get('checkpoint_model'):
        raise ValueError('native checkpoint pointers are incomplete')
    model=Path(progress['checkpoint_model']);index=Path(progress['checkpoint_replay_index'])
    model_bytes=model.read_bytes();index_bytes=index.read_bytes()
    artifact=json.loads(model_bytes);replay=json.loads(index_bytes)
    if artifact.get('updates')!=progress['updates']:raise ValueError('model updates mismatch')
    if progress.get('durable_version',progress['version'])!=progress['version']:
        raise ValueError('native checkpoint version mismatch')
    if replay.get('schema')!='paisho-gen5-replay-index-v1' or replay.get('rules')!='skud-pai-sho-gen5-v1' or not isinstance(replay.get('rows'),list):
        raise ValueError('invalid native replay index')
    count=0
    for row in replay['rows']:
        source=row.get('source',{});indices=row.get('indices')
        if (not isinstance(source.get('path'),str) or not source['path']
            or not isinstance(source.get('sha256'),str) or len(source['sha256'])!=64
            or any(c not in '0123456789abcdef' for c in source['sha256'])
            or not isinstance(indices,list) or any(type(i) is not int or i<0 for i in indices)
            or not isinstance(row.get('lane'),str)):
            raise ValueError('invalid native replay row')
        count+=len(indices)
    if count!=progress['replay_positions']:raise ValueError('native replay position count mismatch')
    return dict(model=model,index=index,model_bytes=model_bytes,index_bytes=index_bytes,
                model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                index_sha256=hashlib.sha256(index_bytes).hexdigest(),rows=replay['rows'])


def prepare(campaign, output):
    campaign=Path(campaign).resolve();output=Path(output).resolve()
    config=json.loads((campaign/'config.json').read_text())
    progress_path=campaign/'training'/('durable-progress.json' if config.get('checkpoint_seconds',0)>0 else 'progress.json')
    original=progress_path.read_bytes();progress=json.loads(original)
    plan=json.loads((campaign/'plan.json').read_text())
    binary_hash=plan['hashes']['bin/paisho-gen5']
    prior=json.loads(Path(config['resume_progress']).read_text()) if config.get('resume_progress') else {}
    previous_completed=prior.get('completed',0)
    if digest(campaign/'bin/paisho-gen5')!=binary_hash:raise ValueError('frozen executable changed')
    capacity=progress['replay_positions']
    checkpoint=native_checkpoint(progress)
    if checkpoint is None and progress['replay_bytes']>=config.get('replay_max_bytes',24*1024**3):
        raise ValueError('byte-limited replay needs native reconstruction')
    rows=[]
    if (campaign/'training.log').exists():
        with (campaign/'training.log').open() as stream:
            for line in stream:
                if not line.endswith('\n'):break
                if '"lane":' in line:rows.append(json.loads(line))
    seen_next_id=max([r['id'] for r in rows],default=-1)+1
    if config.get('checkpoint_seconds',0)>0:
        rows=rows[:progress['completed']-previous_completed]
    if len(rows)==progress['completed']-previous_completed-1:
        rows.append(json.loads((campaign/'training/games'/f"game-{progress['last_game_id']:07}.json").read_text()))
    if len(rows)!=progress['completed']-previous_completed:raise ValueError('receipt/progress count mismatch')
    if not rows and checkpoint is None:raise ValueError('no new durable receipt to resume')
    if rows and rows[-1]['id']!=progress['last_game_id']:raise ValueError('last receipt mismatch')
    if rows and rows[-1]['model_version']!=progress['version']:raise ValueError('model version mismatch')
    retained=deque(checkpoint['rows'] if checkpoint else []);count=capacity if checkpoint else 0
    seats=progress.get('seat_results',prior.get('seat_results',{}))
    if checkpoint is None and config.get('replay_index'):
        retained.extend(json.loads(Path(config['replay_index']).read_text())['rows'])
        count=sum(len(r['indices']) for r in retained)
    for row in rows:
        n=row['eligible_examples']
        if n and checkpoint is None:
            retained.append({'source':{'path':str(campaign/'training/games'/row['targets_file']),
                                      'sha256':row['targets_sha256']},
                             'indices':list(range(n)),'lane':row['lane']})
            count+=n
            while count>capacity:
                remove=min(count-capacity,len(retained[0]['indices']))
                retained[0]['indices']=retained[0]['indices'][remove:];count-=remove
                if not retained[0]['indices']:retained.popleft()
        if 'seat_results' not in progress and row['lane'] not in ('Selfplay', 'Reanalysis'):
            totals=seats.setdefault(row['lane'],dict(games=0,wins=0,losses=0,draws=0,unresolved_seat=0))
            totals['games']+=1
            outcome=row['outcome']
            if outcome=='Draw':totals['draws']+=1
            elif outcome in ('Win(Host)','Win(Guest)'):
                seat=candidate_seat(row,config['seed'],binary_hash)
                if seat is None:raise ValueError('unidentified candidate seat')
                totals['wins' if outcome==f'Win({seat})' else 'losses']+=1
    if count!=capacity:raise ValueError('recent receipts do not cover the retained replay')
    if checkpoint is None:
        for row in retained:
            if digest(Path(row['source']['path']))!=row['source']['sha256']:raise ValueError('target hash mismatch')
    model=checkpoint['model'] if checkpoint else Path(progress['checkpoint_model']) if progress.get('checkpoint_model') else campaign/'training/models'/f"model-{progress['version']:07}.json"
    if json.loads(model.read_text())['updates']!=progress['updates']:raise ValueError('model updates mismatch')
    history=campaign/'training/history'
    next_history=max(prior.get('next_history_index',0),progress.get('next_history_index',0),max([int(p.name.split('-')[1]) for p in history.iterdir() if p.name.startswith('sweep-')],default=-1)+1)
    progress.update(next_game_id=max(prior.get('next_game_id',0),progress.get('next_game_id',0),seen_next_id),next_history_index=next_history,
                    seat_results=seats,previous_campaign=str(campaign))
    if progress_path.read_bytes()!=original:raise ValueError('training changed during capture; pause required')
    if checkpoint and (digest(checkpoint['model'])!=checkpoint['model_sha256'] or digest(checkpoint['index'])!=checkpoint['index_sha256']):
        raise ValueError('native checkpoint changed during capture')
    output.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    write('resume-progress.json',progress)
    if checkpoint:
        (output/'replay.index.json').write_bytes(checkpoint['index_bytes'])
        model=output/'model.json';model.write_bytes(checkpoint['model_bytes'])
        if digest(model)!=checkpoint['model_sha256'] or digest(output/'replay.index.json')!=checkpoint['index_sha256']:
            raise ValueError('native checkpoint copy hash mismatch')
    else:
        write('replay.index.json',{'schema':'paisho-gen5-replay-index-v1','rules':'skud-pai-sho-gen5-v1','rows':list(retained)})
    write('receipt.json',{'model':str(model),'model_sha256':digest(model),'version':progress['version'],
                          'updates':progress['updates'],'completed':progress['completed'],
                          'replay_positions':count,'replay_sources':len(retained),'native_state_exact':checkpoint is not None,
                          'replay_index_sha256':digest(output/'replay.index.json'),
                          'source_progress_sha256':hashlib.sha256(original).hexdigest(),
                          'source_checkpoint_model':str(checkpoint['model']) if checkpoint else None,
                          'source_checkpoint_replay_index':str(checkpoint['index']) if checkpoint else None,
                          'payloads_reloaded_by_preparation':checkpoint is None,
                          'note':'Native durable weights, FIFO and learning state copied exactly; the native importer verifies referenced payload hashes. In-flight search trees are not resumed.' if checkpoint else 'Durable model and retained examples recovered; in-flight games and transient RNG/tree state are not resumed.'})
    return output
