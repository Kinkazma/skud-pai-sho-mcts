import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paisho_gen5_timeline import EloTimeline


class TimelineTests(unittest.TestCase):
    def report(self, root, index, reference, published, wins=3, draws=2, losses=1, unknown=2):
        directory=root/'training/history'/f'sweep-{index:04}'/reference
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'plan.json').write_text(json.dumps({'candidate':f'model-{index}',
            'reference_sha256':reference,'options':{'simulations':64,'decisions':800}}))
        (directory.parent/'assessment.json').write_text(json.dumps({'version':100+index}))
        path=directory/'report.json'
        path.write_text(json.dumps(dict(wins=wins,draws=draws,losses=losses,unknown=unknown)))
        os.utime(path,(published,published))
        return path

    def test_real_observations_separate_references_without_carry_or_fake_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.report(root,0,'initial',10000)
            self.report(root,1,'initial',18000,wins=0,draws=0,losses=0,unknown=8)
            self.report(root,1,'gen3-1',19000,wins=0,draws=0,losses=6)
            reader=EloTimeline(root);rows=reader.read(3)['measurements']
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[0]['version'],100)
            self.assertAlmostEqual(rows[0]['elo'],88.73949984654254)
            self.assertIsNone(rows[1]['elo'])
            self.assertLess(rows[2]['elo'],0)
            self.assertEqual(rows[2]['reference'],'gen3-1')
            with patch('paisho_gen5_timeline.stamp',side_effect=AssertionError('cache miss')):
                self.assertEqual(reader.read(3)['measurements'],rows)

    def test_resume_deduplicates_copies_and_keeps_original_publication_time(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            self.report(old,0,'initial',10000)
            self.report(new,0,'initial',15000)
            self.report(new,1,'initial',18000)
            prior=new/'resume.json';prior.write_text(json.dumps({'previous_campaign':str(old)}))
            (new/'config.json').write_text(json.dumps({'resume_progress':str(prior)}))
            rows=EloTimeline(new).read(3)['measurements']
            self.assertEqual([r['published'] for r in rows],[10000,18000])

    def test_new_report_and_late_assessment_are_picked_up(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reader=EloTimeline(root)
            self.assertEqual(reader.read(2)['measurements'],[])
            path=self.report(root,0,'initial',10000)
            (path.parent.parent/'assessment.json').unlink()
            reader.next_read=0
            self.assertIsNone(reader.read(2)['measurements'][0]['version'])
            (path.parent.parent/'assessment.json').write_text('{"version":1234}')
            reader.next_read=0
            self.assertEqual(reader.read(2)['measurements'][0]['version'],1234)


if __name__=='__main__':unittest.main()
