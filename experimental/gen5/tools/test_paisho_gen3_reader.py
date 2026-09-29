import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paisho_gen3_controls import Gen3Controls
from paisho_gen3_reader import Gen3Reader


class ReaderTests(unittest.TestCase):
    def fixture(self, root, started=1000):
        state=root/'control.json';campaign=root/'run';training=campaign/'training';(training/'games').mkdir(parents=True)
        state.write_text(json.dumps(dict(state='running',generation='3.2',ready=True,campaign=str(campaign),started=started,end=10000)))
        (campaign/'config.json').write_text(json.dumps({'replay_ratio':4}))
        (campaign/'plan.json').write_text(json.dumps({'generation':'3.2'}))
        (training/'progress.json').write_text(json.dumps({'updates':200,'completed':1}))
        return Gen3Reader(Gen3Controls(state)),training
    def row(self,id=0,stamp=1020,historical=True,seat='Guest',outcome='Win(Guest)'):
        return dict(id=id,budget=64,published_unix_seconds=stamp,historical=historical,candidate_seat=seat,outcome=outcome,termination='rules-terminal',decisions=20,learned=50)
    def test_short_rate_window_incremental_seats_and_hourly_bins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,t=self.fixture(root);journal=t/'receipts.jsonl'
            journal.write_text(json.dumps(self.row())+'\n')
            a=r.read(now=1030)['dashboard'];self.assertEqual(a['terminal_per_second'],0)
            self.assertEqual(a['window_seconds'],10);self.assertEqual(a['totals']['Historical']['wins'],1)
            self.assertEqual(a['bins'][0]['start'],0);self.assertEqual(a['totals']['Historical']['fresh'],10)
            self.assertEqual(a['totals']['Historical']['replay'],40)
            self.assertEqual(r.read(now=1040)['dashboard']['totals']['Historical']['games'],1)
            with journal.open('a') as f:f.write(json.dumps(self.row(1,1042,False))+'\n')
            b=r.read(now=1045)['dashboard'];self.assertEqual(b['totals']['Selfplay']['games'],1)
            self.assertEqual(r.read(now=5000)['dashboard']['last_hour'],{})
    def test_partial_journal_line_not_dropped_and_no_model_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,t=self.fixture(Path(tmp));text=json.dumps(self.row());(t/'receipts.jsonl').write_text(text[:20])
            self.assertEqual(r.read(now=1030)['dashboard']['totals'],{})
            with (t/'receipts.jsonl').open('a') as f:f.write(text[20:]+'\n')
            original=Path.open
            def guarded(p,*a,**kw):
                self.assertNotIn('model',p.name);self.assertNotIn('targets',p.name)
                return original(p,*a,**kw)
            with patch.object(Path,'open',guarded):self.assertEqual(r.read(now=1040)['dashboard']['totals']['Historical']['games'],1)
    def test_resume_preserves_totals_but_rate_uses_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,t=self.fixture(root);(t/'receipts.jsonl').write_text(json.dumps(self.row())+'\n');r.read(now=1030)
            second=root/'resume';(second/'training').mkdir(parents=True);(second/'plan.json').write_text(json.dumps({'generation':'3.2'}));(second/'config.json').write_text('{}')
            state=json.loads(r.controls.path.read_text());state.update(campaign=str(second),campaign_history=[str(t.parent),str(second)],started=2000);r.controls.path.write_text(json.dumps(state))
            (second/'training/receipts.jsonl').write_text(json.dumps(self.row(0,2010))+'\n')
            d=r.read(now=2030)['dashboard'];self.assertEqual(d['totals']['Historical']['games'],2);self.assertEqual(d['terminal_per_second'],0)
    def test_elo_is_only_from_frozen_assessments_and_distinct_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,t=self.fixture(root);(t/'receipts.jsonl').write_text(json.dumps(self.row())+'\n')
            self.assertEqual(r.read(now=1030)['dashboard']['evaluated_versions'],0)
            a=root/'assessment.json';a.write_text(json.dumps({'generation':'3.2','published_unix_seconds':1035,'reports':[{'budget':64,'model_sha256':'abc','conditional_internal_elo':12},{'budget':128,'model_sha256':'abc','conditional_internal_elo':None}]}))
            s=json.loads(r.controls.path.read_text());s.update(assessment=str(a),assessment_history=[str(a)]);r.controls.path.write_text(json.dumps(s))
            d=r.read(now=1040)['dashboard'];self.assertEqual(d['evaluated_versions'],1);self.assertEqual(len(d['evaluations']),2)
    def test_online_elo_separates_heuristic_budgets_references_and_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,t=self.fixture(Path(tmp))
            def row(i,budget=64,outcome='Win(Guest)',reference='frozen',version=20):
                return self.row(i,1020+i,outcome=outcome)|dict(opponent='Gen3.1',budget=budget,reference_sha256=reference,collector_sha256='model'+str(version),collector_version=version,rules='v2')
            rows=[self.row(),row(1),row(2,outcome='Win(Host)',version=30),row(3,budget=128),row(4,reference='other'),row(5)|dict(termination='repetition',outcome='Ongoing')]
            (t/'receipts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            d=r.read(now=1030)['dashboard'];self.assertEqual(d['totals']['Historical']['games'],1)
            self.assertEqual(d['totals']['Gen3.1']['games'],5)
            self.assertEqual(d['evaluated_versions'],0)
            groups=d['online_evaluations'];self.assertEqual(len(groups),3)
            g=next(x for x in groups if x['budget']==64 and x['reference_sha256']=='frozen')
            self.assertEqual((g['wins'],g['losses'],g['unknown']),(1,1,1))
            self.assertEqual(g['conditional_internal_elo'],0)
            self.assertEqual((g['version_min'],g['version_max']),(20,30))

    def test_three_frozen_generations_stay_separate_even_with_shared_weight_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,t=self.fixture(Path(tmp))
            rows=[self.row(i,1020+i)|dict(opponent=g,reference_sha256='shared',collector_sha256='model',collector_version=30,rules='v2') for i,g in enumerate(['Gen3.1','Gen3.2','Gen3.3'])]
            (t/'receipts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            d=r.read(now=1040)['dashboard']
            self.assertEqual(set(d['totals']),{'Gen3.1','Gen3.2','Gen3.3'})
            self.assertEqual({x['reference'] for x in d['online_evaluations']},{'Gen3.1','Gen3.2','Gen3.3'})

    def test_hourly_bins_combine_ten_minute_intervals_and_split_next_hour(self):
        with tempfile.TemporaryDirectory() as tmp:
            r,t=self.fixture(Path(tmp))
            rows=[self.row(i,stamp)|dict(opponent='Gen3.1',reference_sha256='fixed',collector_sha256='model',collector_version=i,rules='v2') for i,stamp in enumerate([1200,1800,3599,3600])]
            (t/'receipts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            d=r.read(now=3610)['dashboard'];self.assertEqual(d['bin_seconds'],3600)
            self.assertEqual([(b['start'],b['games']) for b in d['bins']],[(3600,1),(0,3)])
            self.assertEqual([(b['start'],b['games']) for b in d['online_evaluations']],[(3600,1),(0,3)])
            self.assertEqual(d['window_seconds'],600)

    def test_next_generation_does_not_inherit_old_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,t=self.fixture(root);(t/'receipts.jsonl').write_text(json.dumps(self.row())+'\n');r.read(now=1030)
            s=json.loads(r.controls.path.read_text());s.update(generation='3.3');r.controls.path.write_text(json.dumps(s))
            self.assertEqual(r.read(now=1040)['dashboard']['totals'],{})

if __name__=='__main__':unittest.main()
