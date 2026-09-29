"""Compact chronology from saved native reports and independent rule rereads."""
import json
from pathlib import Path
import sys


def choices(read):
    return {r['key'] for r in read['choices'] if r['raw']}


def main(reread, output):
    reads=json.loads((reread/'readings.json').read_text())
    result={'scope':'post-trial union; training/retention diagnostic, not independent strength',
            'reread':json.loads((reread/'summary.json').read_text()),'arms':{}}
    for arm in sorted({r['arm'] for r in reads}):
        selected=[r for r in reads if r['arm']==arm]
        initial=next(r for r in selected if Path(r['path']).name=='actor-000.json')
        first=choices(initial);ever=set(first);last=set(first);actors=[]
        for r in selected:
            name=Path(r['path']).name
            if not name.startswith('actor-'):continue
            current=choices(r)
            actors.append({'model':name,'raw':len(current),'new_since_initial':sorted(current-first),
                           'missing_initial':sorted(first-current),'lost_since_previous':sorted(last-current),
                           'missing_previously_acquired':sorted((ever-first)-current)})
            ever|=current;last=current
        run=Path(initial['path']).parent
        report=json.loads((run/'report.json').read_text())
        expected=report['initial_publication']['accepted_identity'];cycles=[]
        for c in report['cycles']:
            assert c['old_actor']==expected
            assert all(r['actor']==expected for r in c['receipts'])
            expected=c['actor']
            p=c['publication']
            record={'cycle':c['cycle'],'fresh':c['fresh'],'learned':c['learned'],
                    'recall':c['durable_recall'],'changed':c['actor_weights_changed'],
                    'branch':p['accepted_branch'],'fraction':p['accepted_fraction'],
                    'scores':{},'timing':c['timing'],'frozen_wdlu':c['frozen_games']['wdlu'],
                    'structured_positions':c['recall'].get('structured',{}).get('positions',0)}
            for stage in ['scores_actor','scores_learner','scores_unconsolidated']:
                record['scores'][stage]={panel:{k:sum(c[stage][panel][k]) for k in ['raw','coupled']}
                                        for panel in ['primary','validation']}
            cycles.append(record)
        result['arms'][arm]={'run':str(run),'active_seconds':report['active_seconds'],
            'loading_seconds_excluded':report['loading_seconds_excluded'],'actor_data_chain_checked':True,
            'initial_wins':len(first),'actors':actors,'cycles':cycles,
            'fresh':sum(c['fresh'] for c in cycles),'presentations':sum(c['learned'] for c in cycles),
            'recalls':sum(c['recall'] for c in cycles),
            'auxiliary_training_initial':initial['auxiliary'],
            'auxiliary_training_final':next(r['auxiliary'] for r in reversed(selected) if Path(r['path']).name.startswith('actor-'))}
    output.write_text(json.dumps(result,indent=2)+'\n')
    for arm,r in result['arms'].items():
        print(arm,r['fresh'],r['presentations'],r['recalls'],[a['raw'] for a in r['actors']],r['active_seconds'])


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
