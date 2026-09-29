import json
from pathlib import Path
import tempfile
import unittest
from paisho_gen5_seats import candidate_seat, describe_game, legacy_seat, SeatResults, LEGACY_SEAT_BINARY


class SeatTests(unittest.TestCase):
    def test_explicit_seat_and_legacy_schedule_are_never_guessed_for_other_binaries(self):
        self.assertEqual(candidate_seat({'id':3,'candidate_seat':'Guest'},None,'other'),'Guest')
        self.assertIsNone(candidate_seat({'id':3},95717,'other'))
        self.assertEqual(candidate_seat({'id':3},95717,LEGACY_SEAT_BINARY),legacy_seat(95717,3))

    def test_winner_is_model_relative_in_both_seats_and_selfplay_hides_seat_outcome(self):
        for seat in ['Host','Guest']:
            other='Guest' if seat=='Host' else 'Host'
            row={'id':1,'lane':'Historical','candidate_seat':seat,'outcome':f'Win({seat})'}
            self.assertEqual(describe_game(row,0,None)['result_label'],'Gen5 gagne')
            self.assertEqual(describe_game(dict(row,outcome=f'Win({other})'),0,None)['result_label'],'Gen3.1 gagne')
        for outcome in ['Win(Host)','Win(Guest)','Draw']:
            view=describe_game({'id':1,'lane':'Selfplay','outcome':outcome},None,None)
            self.assertEqual(view,{'gen5_seat':'Les deux','result_label':'Terminée'})
        self.assertEqual(describe_game({'id':1,'lane':'Historical','outcome':'Draw'},None,None)['result_label'],'Partie nulle')

    def test_incremental_journal_preserves_partial_lines_and_progress_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'training.log'
            rows=[{'id':0,'lane':'Historical','candidate_seat':'Guest','outcome':'Win(Guest)'},
                  {'id':1,'lane':'Selfplay','outcome':'Win(Host)'},
                  {'id':2,'lane':'Historical','candidate_seat':'Guest','outcome':'Win(Host)'},
                  {'id':3,'lane':'Historical','candidate_seat':'Host','outcome':'Draw'}]
            lines=[json.dumps(r,separators=(',',':')).encode()+b'\n' for r in rows]
            p.write_bytes(b''.join(lines[:3])+lines[3][:-3])
            counts=SeatResults(p);counts.update(1,0,None)
            self.assertEqual(counts.lanes['Historical']['wins'],1)
            counts.update(4,0,None)
            self.assertEqual(counts.processed,3)
            with p.open('ab') as f:f.write(lines[3][-3:])
            counts.update(4,0,None)
            result=counts.snapshot({'Historical':{'games':3}})['Historical']
            self.assertEqual((result['wins'],result['losses'],result['draws']),(1,1,1))
            self.assertTrue(result['ready'])
            self.assertEqual([r['id'] for r in counts.recent['Historical']],[0,2,3])
            self.assertEqual(counts.last_decisive['Historical']['id'],2)
            self.assertFalse(counts.snapshot({'Historical':{'games':4}})['Historical']['ready'])
            before=counts.offset;counts.update(4,0,None);self.assertEqual(counts.offset,before)
            with self.assertRaisesRegex(ValueError,'backwards'):counts.update(3,0,None)

    def test_unknown_seat_does_not_turn_into_an_opponent_win(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'training.log';p.write_text(json.dumps({'id':0,'lane':'Historical','outcome':'Win(Host)'})+'\n')
            counts=SeatResults(p);counts.update(1,0,'unknown')
            row=counts.snapshot({'Historical':{'games':1}})['Historical']
            self.assertEqual(row['losses'],0);self.assertEqual(row['unresolved_seat'],1);self.assertFalse(row['ready'])

if __name__=='__main__':unittest.main()
