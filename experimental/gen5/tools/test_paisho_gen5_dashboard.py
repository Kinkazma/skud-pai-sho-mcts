import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paisho_gen5_dashboard import Gen5Reader, page


class DashboardTests(unittest.TestCase):
    def test_reanalysis_is_not_a_recent_game_or_a_terminal_rate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);games=root/'training/games';games.mkdir(parents=True)
            (root/'config.json').write_text(json.dumps({'case_curriculum':{'losses':10}}))
            (root/'training/progress.json').write_text(json.dumps({'completed':2,'last_game_id':1,'elapsed_seconds':10,'computation_start_elapsed_seconds':0,'lanes':{'Selfplay':{'terminal':1,'games':1},'Reanalysis':{'terminal':0,'games':1}}}))
            rows=[dict(id=0,lane='Selfplay',outcome='Win(Host)',decisions=80,continuation_decisions=20,prefix_decisions=60),dict(id=1,lane='Reanalysis',outcome='Ongoing',reanalysis=True,decisions=65)]
            for row in rows:(games/f"game-{row['id']:07}.json").write_text(json.dumps(row))
            (root/'training.log').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            value=Gen5Reader(root).read()
            self.assertEqual([g['id'] for g in value['games']],[0])
            self.assertEqual(value['games'][0]['continuation_decisions'],20)
            self.assertAlmostEqual(value['gen5_terminal_per_second'],1/10)
            self.assertNotIn('Reanalysis',value['seat_results'])

    def test_startup_preserves_deadline_and_disabled_history_finds_old_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reference=root/'training/history/sweep-0023/initial';reference.mkdir(parents=True)
            prior={'completed':10,'seat_results':{},'elapsed_seconds':60000,
                   'remaining_seconds':0,'next_history_index':24}
            (root/'resume.json').write_text(json.dumps(prior))
            (root/'config.json').write_text(json.dumps({'history_interval':0,'resume_progress':str(root/'resume.json')}))
            (root/'status.json').write_text(json.dumps({'state':'running','end_unix_seconds':8200}))
            (reference/'report.json').write_text(json.dumps({'wins':2,'draws':1,'losses':1}))
            with patch('paisho_gen5_dashboard.time.time',return_value=1000):
                value=Gen5Reader(root).read()
            self.assertTrue(value['initializing'])
            self.assertEqual(value['progress']['remaining_seconds'],7200)
            self.assertEqual(value['internal_evaluation']['index'],23)
            self.assertIsNone(value['gen5_terminal_per_second'])
            self.assertEqual(prior['remaining_seconds'],0)
            content=page().decode()
            for mode in ('selfplay','gen3-replay'):
                self.assertIn("trainingAction('start',2,'"+mode+"')",content)

    def test_late_overnight_evaluation_is_visible_beyond_first_64_sweeps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reference=root/'training/history/sweep-0067/initial';reference.mkdir(parents=True)
            (root/'config.json').write_text(json.dumps({'history_interval':900}))
            (root/'training/progress.json').write_text(json.dumps({'elapsed_seconds':68*900}))
            (reference/'report.json').write_text(json.dumps({'wins':6,'draws':2,'losses':2,'unknown':0}))
            with patch.object(Path,'iterdir',side_effect=AssertionError('archive walk')):
                self.assertEqual(Gen5Reader(root).read()['internal_evaluation']['index'],67)

    def test_bounded_receipts_and_cache_without_opening_models_or_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);games=root/'training/games';games.mkdir(parents=True)
            (root/'config.json').write_text(json.dumps({'threads':10,'secondary_threads':2,'history_interval':900}))
            (root/'training/progress.json').write_text(json.dumps({'last_game_id':90,'completed':91,'elapsed_seconds':10,'computation_start_elapsed_seconds':0,'lanes':{'Selfplay':{'terminal':80},'Historical':{'terminal':1,'games':1}},'secondary_usage':{'reserved_seconds':2,'elapsed_seconds':10}}))
            (games/'game-0000090.json').write_text(json.dumps({'id':90,'outcome':'Win(Host)','decisions':40}))
            (games/'game-0000090.psr').write_text('PSR test')
            evaluation=root/'training/history/sweep-0000/initial'
            evaluation.mkdir(parents=True)
            (evaluation/'plan.json').write_text('{}')
            (evaluation/'game-0000.json').write_text(json.dumps({'score':0.5}))
            (evaluation/'game-0001.json').write_text(json.dumps({'score':None}))
            reader=Gen5Reader(root)
            with patch.object(Path,'glob',side_effect=AssertionError('archive walk')),patch.object(Path,'iterdir',side_effect=AssertionError('archive walk')):
                value=reader.read()
                self.assertEqual(value['gen5_terminal_per_second'],8)  # native counters do not wait for history
                self.assertEqual(value['historical_reserved_capacity_percent'],4)
                self.assertEqual(value['games'][0]['id'],90)
                self.assertEqual(value['evaluation_progress'],{'index':0,'processed':2,'resolved':1,'planned':12,'complete':False})
                with patch('paisho_gen5_dashboard.small_json',side_effect=AssertionError('cache miss')):
                    self.assertIs(reader.read(),value)
            self.assertEqual(reader.psr('90'),b'PSR test')
            with self.assertRaises(FileNotFoundError):reader.psr('../config')

    def test_missing_campaign_stays_preparing_and_has_no_ppo_control(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(Gen5Reader(d).read()['progress'],{})
        content=page().decode()
        self.assertIn('setInterval(refresh,5000)',content)
        self.assertIn('Télécharger',content)
        self.assertNotIn('/api/resume',content)
        self.assertNotIn('const token=',content)

    def test_internal_elo_is_available_before_the_gen3_1_panel_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reference=root/'training/history/sweep-0000/initial';reference.mkdir(parents=True)
            (root/'training/progress.json').write_text(json.dumps({'elapsed_seconds':1000}))
            (reference/'plan.json').write_text('{}')
            report={'wins':3,'draws':2,'losses':3,'unknown':0}
            (reference/'report.json').write_text(json.dumps(report))
            value=Gen5Reader(root).read()
            self.assertEqual(value['assessment'],{})
            self.assertEqual(value['internal_evaluation']['elo'],0)
            self.assertEqual(value['internal_evaluation']['report'],report)
            self.assertFalse(value['evaluation_progress']['complete'])
            report.update(wins=0,draws=0,losses=0,unknown=8)
            (reference/'report.json').write_text(json.dumps(report))
            self.assertIsNone(Gen5Reader(root).read()['internal_evaluation']['elo'])

    def test_evaluation_size_comes_from_its_frozen_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reference=root/'training/history/sweep-0000/initial';reference.mkdir(parents=True)
            (root/'training/progress.json').write_text(json.dumps({'elapsed_seconds':1000}))
            (reference/'plan.json').write_text(json.dumps({'options':{'pairs':50,'decisions':800}}))
            (reference/'game-0099.json').write_text(json.dumps({'score':1.0}))
            progress=Gen5Reader(root).read()['evaluation_progress']
            self.assertEqual(progress['planned'],104)
            self.assertEqual(progress['processed'],1)
            self.assertEqual(progress['resolved'],1)
            (reference.parent/'assessment.json').write_text(json.dumps({'scope':'internal-only'}))
            progress=Gen5Reader(root).read()['evaluation_progress']
            self.assertEqual(progress['planned'],100)
            self.assertTrue(progress['complete'])

    def test_resume_seeds_totals_without_recounting_the_previous_journal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir()
            prior={'completed':10,'lanes':{'Historical':{'games':2,'terminal':2}},'seat_results':{
                'Historical':{'games':2,'wins':1,'losses':1,'draws':0,'unresolved_seat':0}}}
            (root/'resume.json').write_text(json.dumps(prior))
            (root/'config.json').write_text(json.dumps({'resume_progress':str(root/'resume.json')}))
            reader=Gen5Reader(root);value=reader.read()
            self.assertEqual(value['progress']['completed'],10)
            self.assertTrue(value['seat_results']['Historical']['ready'])
            (root/'training.log').write_text(json.dumps({'id':11,'lane':'Historical','candidate_seat':'Guest','outcome':'Win(Guest)'})+'\n')
            (root/'training/progress.json').write_text(json.dumps({'completed':11,'lanes':{'Historical':{'games':3,'terminal':3}}}))
            reader.next_status=0;value=reader.read()
            self.assertEqual(value['seat_results']['Historical']['wins'],2)
            self.assertEqual(reader.seat_results.processed,11)

    def test_native_resume_without_display_totals_still_reads_new_games(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir()
            prior={'completed':10,'lanes':{'Historical':{'games':2,'terminal':2}}}
            (root/'resume.json').write_text(json.dumps(prior))
            (root/'config.json').write_text(json.dumps({'resume_progress':str(root/'resume.json')}))
            reader=Gen5Reader(root);value=reader.read()
            self.assertEqual(value['progress']['completed'],10)
            self.assertFalse(value['seat_results']['Historical']['ready'])
            self.assertEqual(value['seat_results']['Historical']['wins'],0)
            (root/'training.log').write_text(json.dumps({'id':11,'lane':'Historical','candidate_seat':'Guest','outcome':'Win(Guest)'})+'\n')
            (root/'training/progress.json').write_text(json.dumps({'completed':11,'lanes':{'Historical':{'games':3,'terminal':3}}}))
            reader.next_status=0;value=reader.read()
            self.assertEqual(value['seat_results']['Historical']['games'],1)
            self.assertEqual(value['seat_results']['Historical']['wins'],1)
            self.assertFalse(value['seat_results']['Historical']['ready'])
            self.assertEqual(reader.seat_results.processed,11)

if __name__=='__main__':unittest.main()
