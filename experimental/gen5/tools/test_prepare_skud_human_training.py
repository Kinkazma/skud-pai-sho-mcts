import json
from pathlib import Path
import tempfile
import unittest
import prepare_skud_human_training as prepare

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'target/release/examples/convert_site_record'
NOTATION = '0H.R,W,K,B;0G.R,W,K,B;1G.R3(0,-8);1H.R3(0,8);2G.R4(8,0);2H.W5(-8,0)\n'


class CorpusTests(unittest.TestCase):
    def game(self, identity=1, winner='Alice'):
        return dict(game_id=identity, host='Alice', guest='Bob', winner=winner, result_id=9,
                    tournaments=[dict(gameWinnerUsername=winner)], options=[],
                    notation_path=f'notations/{identity}.txt', notation_sha256=prepare.sha(NOTATION.encode()),
                    notation_status='present_unvalidated')

    def test_declared_identity_and_conflicts(self):
        self.assertEqual(prepare.declared_side(self.game()), 'H')
        self.assertEqual(prepare.declared_side(self.game(winner='Bob')), 'G')
        with self.assertRaises(ValueError):
            prepare.declared_side(self.game(winner='unknown'))
        game = self.game(); game['tournaments'][0]['gameWinnerUsername'] = 'Bob'
        with self.assertRaises(ValueError):
            prepare.declared_side(game)

    @unittest.skipUnless(BINARY.exists(), 'build native convert_site_record example first')
    def test_resignation_is_legal_prefix_dedup_and_conflict(self):
        for contradictory in [False, True]:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); source = root/'source'; source.mkdir(); (source/'notations').mkdir()
                games = [self.game(), self.game(2, 'Bob' if contradictory else 'Alice')]
                for game in games:
                    (source/game['notation_path']).write_text(NOTATION)
                (source/'games.json').write_text(json.dumps(games))
                result = prepare.prepare(source, root/'result', BINARY, 1)
                self.assertEqual(result['accepted_unique_games'], 0 if contradictory else 1)
                records = json.loads((root/'result/records.json').read_text())
                self.assertTrue(all(r['facts']['outcome']=='ongoing' for r in records))
                if not contradictory:
                    sidecars = list((root/'result/accepted').glob('*.outcome.json'))
                    self.assertEqual(len(sidecars), 1)
                    self.assertEqual(len(json.loads(sidecars[0].read_text())['source_records']), 2)
                with self.assertRaises(FileExistsError):
                    prepare.prepare(source, root/'result', BINARY, 1)

    @unittest.skipUnless(BINARY.exists(), 'build native convert_site_record example first')
    def test_source_hash_and_unsupported_options(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'notations').mkdir(); (root/'replayed').mkdir()
            game=self.game(); (root/game['notation_path']).write_text(NOTATION)
            game['options']=['variant']
            self.assertEqual(prepare.convert_one(game,root,root,BINARY)['reason'], 'unsupported_options')
            game['notation_sha256']='0'*64
            with self.assertRaises(ValueError): prepare.convert_one(game,root,root,BINARY)

if __name__ == '__main__':
    unittest.main()
