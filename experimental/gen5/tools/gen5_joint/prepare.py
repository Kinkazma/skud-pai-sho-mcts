"""Freeze larger legal-source experiment before any candidate learning."""
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base=Path('benchmarks/results/gen5-r3-r5-integration-2026-09-26')
    old=Path('benchmarks/results/gen5-r2-native-2026-09-26/input/plan.json')
    r2=json.loads(old.read_text())
    human=Path('training-runs/micro-gen5-ui-20260919-143403-44f847/human-dataset.json')
    games=json.loads(human.read_text())['games']
    used_held={r['group'] for r in r2['rows'] if r['held_out']}
    def split(g,held):
        if held:
            return 'historical_control' if g in used_held else 'final'
        return 'dev' if int(hashlib.sha256(('joint-dev-v1:'+g).encode()).hexdigest()[:8],16)%5==0 else 'train'
    sources=[]
    for g in games:
        group=g['split_identity_sha256']; s=split(group,g['held_out'])
        if s=='historical_control': continue
        item=g['originals'][0]
        assert sha(item['path'])==item['sha256']
        sources.append({'psr':item['path'],'sha256':item['sha256'],'group':group,'split':s,'kind':'human'})
    seen=set()
    for r in r2['rows']:
        if r['held_out'] or r['psr'] in seen: continue
        seen.add(r['psr']); sources.append({k:r[k] for k in ['psr','sha256','group']}|
            {'split':split(r['group'],False),'kind':'r1'})
    legacy=Path('benchmarks/results/gen5-r1-geometry-2026-09-26/frozen-lots-v3/corrected-legacy')
    for r in json.loads((legacy/'census/roots.json').read_text()):
        if r['held_out']: continue
        p=legacy/'census'/r['psr']
        sources.append({'psr':str(p.resolve()),'sha256':r['sha256'],'group':r['group'],
                        'split':split(r['group'],False),'kind':'r1'})
    branches=legacy/'branches/branches.jsonl'
    plan={'schema':'gen5-joint-consequences-plan-v1','actor':r2['actor'],'actor_sha256':r2['actor_sha256'],
          'human_manifest_sha256':sha(human),'sources':sources,
          'legacy_roots':str((legacy/'census/roots.json').resolve()),
          'legacy_roots_sha256':sha(legacy/'census/roots.json'),
          'legacy_branches':str(branches.resolve()),'legacy_branches_sha256':sha(branches),
          'human_decisions':'midpoint and last recorded decision, zero-based prefixes',
          'seeds':[17,29,43], 'arms':['dense-policy','dense-joint','messages-policy','messages-joint'],
          'steps_per_phase':4096,'read_steps':[0,1024,4096], 'rate':.0003,
          'auxiliary_counts_weight':.2,'auxiliary_events_weight':.2,'unknown_policy_kl_weight':.1,
          'phase_B_recall':.5,'rare_root_sampling_fraction':.5,
          'final_controls_exclude_R2_groups':True,'activation':False}
    target=base/'plan.json'
    with target.open('x') as f: json.dump(plan,f,indent=2);f.write('\n')
    print(len(sources),'source references frozen')


if __name__=='__main__':main()
