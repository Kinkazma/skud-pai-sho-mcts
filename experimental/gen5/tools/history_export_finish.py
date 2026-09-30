#!/usr/bin/env python3
"""Finish FIFO and dependency pointers after the bounded archive export."""
import argparse,concurrent.futures,json,sqlite3,time
from pathlib import Path
from export_history import atomic,export_one,sha

def fifo(state,journal,workers):
    jobs=json.loads((state/'fifo-jobs.private.json').read_text())
    mapping={};started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i in range(0,len(jobs),64):
            items=[(Path(a),Path(b),c,d) for a,b,c,d in jobs[i:i+64]]
            for old,dest,oldsha,newsha,n,keys in pool.map(export_one,items):mapping[old]={'path':dest,'sha256':newsha}
            if i%1024==0:print(json.dumps({'fifo':i+len(items),'seconds':time.monotonic()-started}),flush=True)
    private=json.loads(journal.with_suffix('.state.json').read_text());root=Path(private['root'])
    source=json.loads(Path(private['config']).read_text());index=json.loads(Path(source['replay_index']).read_text())
    for row in index['rows']:
        x=mapping[row['source']['path']];row['source']={'path':str(Path(x['path']).relative_to(root)),'sha256':x['sha256']}
    atomic(state/'replay.index.json',json.dumps(index,separators=(',',':')).encode())
    atomic(journal.with_suffix('.fifo.json'),json.dumps(mapping).encode())
    (state/'fifo-jobs.private.json').unlink()
    print(json.dumps({'fifo_files':len(mapping),'positions':sum(len(r['indices']) for r in index['rows'])}),flush=True)

def metadata(state,journal):
    private=json.loads(journal.with_suffix('.state.json').read_text());root=Path(private['root'])
    source=json.loads(Path(private['progress']).read_text());p=json.loads((state/'progress.partial.json').read_text())
    db=sqlite3.connect(journal);lookup={s:d for s,d in db.execute('select source,destination from files')};db.close()
    for lane in ['proofs','ordinary']:
        cursor=source['durable_recall']['coverage'][lane]['cursor'];raw=Path(cursor['manifest']).read_bytes()
        if sha(raw)!=cursor['sha256']:raise ValueError('Coverage source hash mismatch')
        paths=json.loads(raw)
        rows=[str(Path(lookup[s]).relative_to(root)) if Path(lookup[s]).is_absolute() else lookup[s] for s in paths]
        target=state/f'coverage-{lane}.json';body=json.dumps(rows,separators=(',',':')).encode();atomic(target,body)
        p['durable_recall']['coverage'][lane]['cursor']['sha256']=sha(body)
    p['durable_recall']['next_bundle']=lookup[source['durable_recall']['next_bundle']]
    atomic(state/'progress.partial.json',json.dumps(p,separators=(',',':')).encode())
    print(json.dumps({'coverage_order_preserved':True,'archive_files':len(lookup)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fifo','metadata']);p.add_argument('state',type=Path);p.add_argument('--journal',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.mode=='fifo':fifo(a.state,a.journal,a.workers)
    else:metadata(a.state,a.journal)
