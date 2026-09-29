import json
import os
from pathlib import Path
import tempfile
import unittest
from paisho_gen5_hour import HourResults, complete_end, reverse_lines


class HourTests(unittest.TestCase):
    def row(self, root, game, timestamp, lane='Historical', seat='Guest', outcome='Win(Guest)', budget=None, **extra):
        games=root/'training/games';games.mkdir(parents=True,exist_ok=True)
        row={'id':game,'lane':lane,'candidate_seat':seat,'outcome':outcome,
             'decisions':20,'fresh_used':15,'replay_used':60,'reference_budget':budget,**extra}
        text=json.dumps(row,separators=(',',':')).encode()+b'\n'
        receipt=games/f'game-{game:07}.json';receipt.write_bytes(text)
        os.utime(receipt,(timestamp,timestamp))
        with (root/'training.log').open('ab') as log:log.write(text)
        return text

    def read(self, tracker, now, **kwargs):
        return tracker.update(complete_end(tracker.campaign/'training.log'),1,None,now=now,budget_seconds=1,**kwargs)

    def test_ten_minute_throughput_reopen_and_expiry_excludes_unknown_and_reanalysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for game, timestamp, outcome in [(0,9399,'Win(Host)'),(1,9400,'Draw'),(2,9401,'Draw'),(3,9999,'Ongoing')]:
                self.row(root,game,timestamp,lane='Selfplay',outcome=outcome)
            self.row(root,4,9999,lane='Reanalysis',outcome='Win(Host)')
            for _ in range(2):
                tracker=HourResults(root);self.read(tracker,10000)
                rates=tracker.throughput(10000)
                self.assertTrue(rates['ready'])
                self.assertEqual(rates['terminal'],{'Selfplay':1})
                self.assertEqual(rates['terminal_per_second']['Selfplay'],1/600)
                self.read(tracker,10600)
                self.assertEqual(tracker.throughput(10600)['terminal']['Selfplay'],0)

    def test_ten_minute_window_uses_persisted_launch_time_before_ten_minutes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'started.json').write_text(json.dumps({'unix_seconds':9950}))
            self.row(root,0,9949,lane='Selfplay')
            self.row(root,1,9999,lane='Selfplay')
            tracker=HourResults(root);self.read(tracker,10000)
            rates=tracker.throughput(10000)
            self.assertEqual(rates['seconds'],50)
            self.assertEqual(rates['terminal_per_second']['Selfplay'],.02)

    def test_launch_receipt_arriving_after_reader_uses_thirty_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            tracker=HourResults(root)
            (root/'started.json').write_text(json.dumps({'unix_seconds':9970}))
            self.row(root,0,9969,lane='Selfplay')
            self.row(root,1,9999,lane='Selfplay')
            self.read(tracker,10000)
            rates=tracker.throughput(10000)
            self.assertEqual(rates['seconds'],30)
            self.assertAlmostEqual(rates['terminal_per_second']['Selfplay'],1/30)

    def test_current_ten_minute_bins_do_not_wait_for_ancestry_and_separate_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'current';root.mkdir();old=Path(d)/'old';old.mkdir()
            for i in range(10):self.row(old,i,9900,budget=128)
            (root/'resume.json').write_text(json.dumps({'previous_campaign':str(old)}))
            (root/'config.json').write_text(json.dumps({'resume_progress':str(root/'resume.json')}))
            (root/'started.json').write_text(json.dumps({'unix_seconds':9970}))
            self.row(root,0,9990,budget=8)
            self.row(root,1,9991,seat='Host',outcome='Win(Guest)',budget=32)
            tracker=HourResults(root);self.read(tracker,10000,max_rows=3)
            self.assertTrue(tracker.loading)  # old hourly history is still loading
            bins=tracker.replay_history(10000)
            self.assertTrue(bins['ready'])
            self.assertEqual(len(bins['bins']),2)
            by_budget={r['budget']:r for r in bins['bins']}
            self.assertEqual(by_budget[8]['wins'],1)
            self.assertEqual(by_budget[32]['losses'],1)
            self.assertEqual(by_budget[8]['end']-by_budget[8]['start'],600)
            rates=tracker.throughput(10000)
            self.assertTrue(rates['ready'])
            self.assertAlmostEqual(rates['terminal_per_second']['Historical'],2/30)

    def test_matching_budget_never_merges_generations_or_counts_reanalysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.row(root,0,9990,budget=8,opponent='Gen3.1',reference_identity='a')
            self.row(root,1,9991,budget=8,opponent='Gen3.5',reference_identity='b',outcome='Win(Host)')
            self.row(root,2,9992,budget=8,opponent='Gen3.5',reference_identity='c',outcome='Draw')
            self.row(root,3,9993,lane='Selfplay')
            self.row(root,4,9994,lane='Reanalysis',opponent='Gen3.5',budget=8)
            tracker=HourResults(root);self.read(tracker,10000)
            h=tracker.replay_history(10000)
            self.assertEqual(h['bins'][0]['games'],3)  # preserved aggregate contract
            rows=h['match_bins'];self.assertEqual(sum(r['games'] for r in rows),4)
            first=next(r for r in rows if r['generation']=='Gen3.1')
            self.assertEqual((first['wins'],first['losses']),(1,0))
            fifth=[r for r in rows if r['generation']=='Gen3.5']
            self.assertEqual({r['reference_identity'] for r in fifth},{'b','c'})
            self.assertEqual(sum(r['losses'] for r in fifth),1)
            self.assertEqual(sum(r['draws'] for r in fifth),1)
            self.assertEqual(next(r for r in rows if r['lane']=='Selfplay')['games'],1)

    def test_exact_window_model_seats_draws_and_expiry_without_new_games(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.row(root,0,6399)
            self.row(root,1,6400)  # exactly one hour old: excluded
            self.row(root,2,6401)
            self.row(root,3,7000,seat='Host',outcome='Win(Guest)')
            self.row(root,4,8000,outcome='Draw')
            self.row(root,5,9000,outcome='Ongoing')
            self.row(root,6,9500,lane='Selfplay')
            tracker=HourResults(root);r=self.read(tracker,10000)
            self.assertTrue(r['ready'])
            h=r['lanes']['Historical']
            self.assertEqual([h[k] for k in ('games','terminal','wins','losses','draws')],[4,3,1,1,1])
            self.assertEqual(h['fresh_used'],60)
            self.assertEqual(r['lanes']['Selfplay']['terminal'],1)
            later=self.read(tracker,13000)
            self.assertNotIn('Historical',later['lanes'])
            self.assertEqual(later['lanes']['Selfplay']['games'],1)

    def test_resume_includes_recent_previous_campaign_without_double_counting(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old';new=Path(d)/'new';old.mkdir();new.mkdir()
            self.row(old,0,6300)
            self.row(old,1,6900)
            self.row(new,2,8000,outcome='Draw')
            prior=new/'resume.json';prior.write_text(json.dumps({'previous_campaign':str(old)}))
            (new/'config.json').write_text(json.dumps({'resume_progress':str(prior)}))
            tracker=HourResults(new)
            first=self.read(tracker,10000)
            self.assertTrue(first['ready']);self.assertEqual(first['lanes']['Historical']['games'],2)
            self.assertEqual(self.read(tracker,10000)['lanes'],first['lanes'])

    def test_bounded_loading_partial_lines_and_forward_progress_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.row(root,0,9900);self.row(root,1,9901)
            tracker=HourResults(root)
            self.assertFalse(self.read(tracker,10000,max_rows=1)['ready'])
            self.assertTrue(self.read(tracker,10000)['ready'])
            boundary=complete_end(root/'training.log')
            text=self.row(root,2,9950,outcome='Draw')
            with (root/'training.log').open('r+b') as log:log.truncate(boundary+len(text)-2)
            self.assertEqual(self.read(tracker,10000)['lanes']['Historical']['games'],2)
            with (root/'training.log').open('ab') as log:log.write(text[-2:])
            r=tracker.update(boundary,1,None,now=10000)
            self.assertEqual(r['lanes']['Historical']['games'],2)
            self.assertEqual(self.read(tracker,10000)['lanes']['Historical']['games'],3)

    def test_fixed_hours_keep_old_results_and_split_at_exact_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.row(root,0,3599)
            self.row(root,1,3600,outcome='Draw')
            self.row(root,2,7200,seat='Host',outcome='Win(Guest)')
            tracker=HourResults(root);self.read(tracker,8000)
            history=tracker.history(now=8000)
            self.assertTrue(history['ready'])
            self.assertEqual([r['start'] for r in history['hours']],[7200,3600,0])
            self.assertEqual([r['lanes']['Historical']['games'] for r in history['hours']],[1,1,1])
            self.assertEqual(history['hours'][0]['lanes']['Historical']['losses'],1)
            self.assertTrue(history['hours'][0]['partial'])
            self.read(tracker,20000)
            self.assertEqual(tracker.history(now=20000)['hours'][-1]['lanes']['Historical']['wins'],1)

    def test_new_receipts_before_bootstrap_begins_are_counted_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.row(root,0,9900)
            tracker=HourResults(root)
            self.row(root,1,9901)
            result=self.read(tracker,10000)
            self.assertEqual(result['lanes']['Historical']['games'],2)
            self.assertEqual(tracker.history(10000)['hours'][0]['lanes']['Historical']['games'],2)

    def test_archived_last_durable_receipt_missing_from_stdout_is_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old';new=Path(d)/'new';old.mkdir();new.mkdir()
            self.row(old,0,7000)
            old_end=complete_end(old/'training.log')
            self.row(old,2,7100,lane='Selfplay')
            with (old/'training.log').open('r+b') as log:log.truncate(old_end)
            (old/'training/progress.json').write_text(json.dumps({'completed':2,'last_game_id':2}))
            self.row(new,3,8000)
            prior=new/'resume.json';prior.write_text(json.dumps({'previous_campaign':str(old),'completed':2}))
            (new/'config.json').write_text(json.dumps({'resume_progress':str(prior)}))
            tracker=HourResults(new);result=self.read(tracker,10000)
            self.assertTrue(result['ready'])
            self.assertEqual(result['lanes']['Selfplay']['games'],1)
            self.assertEqual(sum(v['games'] for h in tracker.history()['hours'] for v in h['lanes'].values()),3)

    def test_unknown_seat_is_not_an_opponent_win(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.row(root,0,9900,seat=None)
            r=self.read(HourResults(root),10000)
            self.assertTrue(r['ready'])
            h=r['lanes']['Historical'];self.assertFalse(h['ready'])
            self.assertEqual(h['losses'],0);self.assertEqual(h['unresolved_seat'],1)

    def test_reverse_reader_handles_block_boundaries_and_excludes_partial_tail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log';lines=[str(i).encode()*701 for i in range(100)]
            p.write_bytes(b'\n'.join(lines)+b'\npartial')
            self.assertEqual(list(reverse_lines(p,complete_end(p))),list(reversed(lines)))

    def test_missing_timestamp_reports_unavailable_instead_of_false_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.row(root,0,9900)
            (root/'training/games/game-0000000.json').unlink()
            r=self.read(HourResults(root),10000)
            self.assertFalse(r['ready']);self.assertTrue(r['error'])


if __name__=='__main__':unittest.main()
