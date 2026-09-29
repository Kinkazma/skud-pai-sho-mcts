"""Offline checks for public-corpus parsing, provenance and resumability."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import download_skud_tournaments as scrape


def info(game_id=42, game_type='Skud Pai Sho', options='[]'):
    return '|||'.join(map(str, [game_id, 1, game_type, 'Host', 0, 'Guest', 0,
                              1, options, 'Guest', 1, '2021-04-10', 1093,
                              '', '', ''])).encode()


class Response:
    status = 200
    headers = {'Content-Type': 'text/plain'}

    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return self.raw


class PublicCorpusTests(unittest.TestCase):
    def test_metadata_does_not_invent_historical_ratings(self):
        parsed = scrape.parse_game_info(info(), 42)
        self.assertEqual(parsed['host_rating'], 1093)
        self.assertIsNone(parsed['guest_rating'])
        self.assertIsNone(parsed['ranked_raw'])
        self.assertTrue(parsed['rating_time_semantics'].startswith('unknown'))
        self.assertEqual(parsed['winner'], 'Guest')

    def test_html_encoded_rule_options_are_preserved(self):
        parsed = scrape.parse_game_info(info(options='[&quot;DoubleAccentTiles&quot;,&quot;AncientOasisExpansion&quot;]'), 42)
        self.assertEqual(parsed['options'], ['DoubleAccentTiles', 'AncientOasisExpansion'])

    def test_game_identity_and_variant_are_checked(self):
        for raw in (info(43), info(game_type='Vagabond Pai Sho'),
                    info(options='{}'), b'PHP warning', info() + b'\n' + info()):
            with self.assertRaises(ValueError):
                scrape.parse_game_info(raw, 42)

    def test_index_rejects_json_errors_and_bad_structure(self):
        for value in ({'error': 'temporary failure'}, None, [],
                      {'tournamentList': [None]}, {'tournamentList': [{'id': 1}, {'id': 1}]}):
            with self.assertRaises(ValueError):
                scrape.parse_tournament_index(json.dumps(value).encode())
        self.assertEqual(scrape.parse_tournament_index(b'{"tournamentList":[]}'), {'tournamentList': []})

    def test_empty_and_incomplete_sources_are_distinct(self):
        self.assertEqual(scrape.notation_status(''), 'empty')
        self.assertEqual(scrape.notation_status('0H.R,W,K,B'), 'setup_incomplete')
        self.assertEqual(scrape.notation_status('old format'), 'unrecognized_header')
        self.assertEqual(scrape.notation_status('0G.R,W,K,B;0H.R,W,K,B;'),
                         'present_unvalidated')

    def test_verified_cache_avoids_requests_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = scrape.Archive(Path(directory), 0)
            with patch.object(scrape, 'urlopen', return_value=Response(b'original')) as get:
                self.assertEqual(archive.fetch('raw/a.txt', 'public', {'q': 42}), b'original')
                self.assertEqual(archive.fetch('raw/a.txt', 'public', {'q': 42}), b'original')
                self.assertEqual(get.call_count, 1)
                (Path(directory) / 'raw/a.txt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'cached response changed'):
                    archive.fetch('raw/a.txt', 'public', {'q': 42})
                self.assertEqual(get.call_count, 1)

    def test_invalid_successful_response_is_preserved_and_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = scrape.Archive(Path(directory), 0)
            with patch.object(scrape, 'urlopen', side_effect=[Response(b'PHP warning'), Response(b'{"id":42}')]) as get:
                parsed = archive.validated_fetch('raw/a.json', 'public', {}, json.loads)
                self.assertEqual(parsed, {'id': 42})
                self.assertEqual(get.call_count, 2)
            rejected = list(Path(directory).glob('rejected/*/raw/a.json'))
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0].read_bytes(), b'PHP warning')
            receipt = json.loads(rejected[0].with_suffix('.json.http.json').read_text())
            self.assertEqual(receipt['sha256'], scrape.sha(b'PHP warning'))

    def test_empty_notation_is_archived_but_html_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = scrape.Archive(Path(directory), 0)
            with patch.object(scrape, 'urlopen', return_value=Response(b'\n\n')):
                self.assertEqual(archive.fetch('empty.txt', 'public', allow_empty=True), b'\n\n')
            with patch.object(scrape, 'urlopen', return_value=Response(b'<!doctype html>error')):
                with self.assertRaises(ValueError):
                    archive.fetch('html.txt', 'public', allow_empty=True)

    def test_download_filters_variants_deduplicates_and_writes_verifiable_index(self):
        def response(request, **kwargs):
            parsed = urlparse(request.full_url)
            query = parse_qs(parsed.query)
            if parsed.path.endswith('getCurrentTournaments.php'):
                data = {'tournamentList': [{'id': 1}, {'id': 2}]}
            elif parsed.path.endswith('getTournamentInfo.php'):
                tid = int(query['t'][0])
                data = {'id': tid, 'name': '<unsafe & name>', 'status': 'Completed',
                        'games': [{'gameId': 42, 'gameType': 'Skud Pai Sho'},
                                  {'gameId': 99, 'gameType': 'Vagabond Pai Sho'}]}
            elif parsed.path.endswith('getGameInfoV2.php'):
                self.assertEqual(query['gameId'], ['42'])
                return Response(info())
            elif parsed.path.endswith('getGameNotation.php'):
                self.assertEqual(query['q'], ['42'])
                return Response(b'\n0H.R,W,K,B;0G.R,W,K,B;\n')
            else:
                self.fail(request.full_url)
            return Response(json.dumps(data).encode())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(scrape, 'urlopen', side_effect=response) as get, \
                    patch.object(scrape.time, 'sleep'), patch('builtins.print'):
                summary = scrape.download(output)
                self.assertEqual(get.call_count, 5)
                second = scrape.download(output, workers=4, interval=.25)
                self.assertEqual(get.call_count, 5)
            self.assertTrue(summary['complete'])
            self.assertTrue(second['complete'])
            self.assertEqual(summary['unique_listed_skud_games'], 1)
            game = json.loads((output / 'games.json').read_text())[0]
            self.assertEqual(len(game['tournaments']), 2)
            self.assertFalse(game['training_eligible'])
            self.assertEqual(game['notation_sha256'], scrape.sha((output / game['notation_path']).read_bytes()))
            page = (output / 'index.html').read_text()
            self.assertNotIn('<unsafe & name>', page)
            self.assertIn('&lt;unsafe &amp; name&gt;', page)
            self.assertEqual(len(list((output / 'runs').glob('*.json'))), 2)


if __name__ == '__main__':
    unittest.main()
