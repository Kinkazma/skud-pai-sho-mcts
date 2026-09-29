#!/usr/bin/env python3
"""Read the portable bank's fixed layout to audit/refresh bounded membership.

This command emits a selection manifest; the native builder validates and packs
it into a new immutable bank. It never deletes PSR originals or live bank files.
"""
import argparse,json,struct,hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
from paisho_sequence_retention import Game,Policy,select

def motifs(path,usage=None):
    data=Path(path).read_bytes()
    stride={b'PSSEQ001':278,b'PSSEQ002':478}.get(data[:8])
    if stride is None or len(data)<24:raise ValueError('bank format')
    games,human,count,clusters=struct.unpack_from('<4I',data,8)
    if len(data)<24+count*stride:raise ValueError('truncated bank entries')
    groups=defaultdict(set);wins=defaultdict(set);uses=defaultdict(int);game_uses=defaultdict(int)
    usage=usage or {}
    for i in range(count):
        offset=24+stride*i
        key=struct.unpack_from('<64h',data,offset)
        # Stable coarse locality signature; all coordinates contribute. Similar
        # signatures are retention neighborhoods, not tactical equivalence.
        # V2 geometry changes retrieval, not the existing admission/retention
        # policy. Skip its 200-byte maps when reading the shared entry prefix.
        signature=0
        for bit in range(16):
            projection=sum(x*(1 if ((j+1)*(bit+17)*2654435761 & (1<<27)) else -1) for j,x in enumerate(key))
            signature|=int(projection>=0)<<bit
        source,g,start,end=struct.unpack_from('<QIII',data,offset+256)
        outcome,phase=struct.unpack_from('<bB',data,offset+276)
        m=f'{phase}:{signature:04x}'
        recalled=usage.get((source,start),0);uses[m]+=recalled;game_uses[g]+=recalled
        groups[g].add(m)
        if outcome==1:wins[g].add(m)
    return games,human,groups,wins,dict(uses),dict(game_uses)

def catalog(bank,sources):
    total,human,groups,wins,_,_=motifs(bank)
    if total!=len(sources):raise ValueError('source catalog mismatch')
    return [Game(s['sha256'],s['human'],tuple(sorted(groups[i])),frozenset(wins[i]),last_added=int(s.get('metadata',{}).get('id') or 0)) for i,s in enumerate(sources)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--incoming-bank',type=Path,required=True);p.add_argument('--incoming-sources',type=Path,required=True)
    p.add_argument('--usage',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();old=json.loads(a.sources.read_text());new=json.loads(a.incoming_sources.read_text())
    existing=catalog(a.bank,old);incoming=catalog(a.incoming_bank,new)
    uses={}
    if a.usage:
        usage=json.loads(a.usage.read_text())
        if usage['bank']['sha256']!=hashlib.sha256(a.bank.read_bytes()).hexdigest():raise ValueError('usage belongs to another bank')
        observations={(r['source'],r['decision']):r['uses'] for r in usage['rows']}
        _,_,_,_,uses,game_uses=motifs(a.bank,observations)
        existing=[replace(g,recent_uses=game_uses.get(i,0)) for i,g in enumerate(existing)]
    chosen,report=select(existing,incoming,uses=uses)
    mapping={s['sha256']:s for s in new};mapping.update({s['sha256']:s for s in old if s['human']})
    for s in old:mapping.setdefault(s['sha256'],s)
    result=[mapping[g.identity] for g in chosen]
    with a.output.open('x') as f:json.dump(result,f)
    a.output.with_suffix('.receipt.json').write_text(json.dumps({**report,'bank_sha256':hashlib.sha256(a.bank.read_bytes()).hexdigest(),'incoming_bank_sha256':hashlib.sha256(a.incoming_bank.read_bytes()).hexdigest(),'policy':Policy().__dict__},indent=2))
    print(json.dumps(report))
if __name__=='__main__':main()
