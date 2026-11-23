"""Every human target once per epoch, in shuffled eight-decision zones."""
import hashlib
import json
from paisho_ppo.human import HumanDataset


class ZoneDataset:
    def __init__(self,zone_decisions=8):
        self.base=HumanDataset()
        self.data=self.base.data
        self.recipe={'schema':1,'kind':'human-single-view-eight-decision-zones-v1',
            'base_corpus_identity':self.base.identity,'zone_decisions':zone_decisions,
            'coverage':'every-observed-human-target-once-each-epoch',
            'openings':'every-official-zero-history-opening-each-epoch',
            'context':'complete-observed-prefix','split':'original-source-groups'}
        self.identity=hashlib.sha256(json.dumps(self.recipe,sort_keys=True).encode()).hexdigest()
        self.entries=[]
        for game,e in enumerate(self.base.entries):
            for offset in range(0,e['decisions'],zone_decisions):
                self.entries.append({'source_game':game,'offset':offset,
                    'decisions':min(zone_decisions,e['decisions']-offset),'group':e['group'],'split':e['split']})
        self.splits={name:[i for i,e in enumerate(self.entries) if e['split']==name] for name in ('train','validation')}

    def fetch(self,refs,engine):
        mapped=[]
        for zone,decision in refs:
            e=self.entries[zone]
            if not 0<=decision<e['decisions']:
                raise ValueError('Zone decision out of range')
            mapped.append([e['source_game'],e['offset']+decision])
        return self.base.fetch(mapped,engine)

    def summary(self):
        return {'corpus_identity':self.identity,'base_corpus_identity':self.base.identity,
            'symmetry_duplicates':False,'recipe':self.recipe,
            'splits':{name:{'zones':len(indices),'examples':sum(self.entries[i]['decisions'] for i in indices),
                'zero_history_openings':len(self.base.splits[name])} for name,indices in self.splits.items()}}
