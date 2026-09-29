import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from migrate_psr_rules import migrate

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / 'target/release/examples/revalidate_record'
FIXTURE = (ROOT / 'crates/paisho-core/tests/fixtures/reported-ring-v1.psr').read_text()


def digest(data):
    return hashlib.sha256(data).hexdigest()


class MigrationTests(unittest.TestCase):
    def setUp(self):
        if not CONVERTER.is_file():
            self.fail('build first: cargo build --release -p paisho-core --example revalidate_record')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.input = self.source / 'game.psr'
        self.input.write_text(FIXTURE)
        self.output = self.root / 'converted'

    def run_migration(self):
        return migrate(self.source, self.output, CONVERTER)

    def record(self):
        return next((self.output / 'records').glob('*.psr'))

    def test_real_game_preserves_original_and_corrects_terminal_and_split(self):
        before = self.input.read_bytes()
        result = self.run_migration()
        self.assertEqual(result['converted'], 1)
        self.assertEqual(result['discarded_decisions'], 2)
        self.assertEqual(result['changed_outcomes'], 1)
        target = self.record()
        receipt = json.loads(target.with_suffix('.rules-migration.json').read_text())
        self.assertEqual(receipt['source_outcome'], 'H')
        self.assertEqual(receipt['target_outcome'], 'G')
        self.assertEqual(receipt['target_decisions'], 74)
        self.assertEqual(receipt['split_identity_sha256'], digest(before))
        self.assertEqual(receipt['target_record_sha256'], digest(target.read_bytes()))
        self.assertEqual(self.input.read_bytes(), before)
        self.assertFalse(target.with_suffix('.outcome.json').exists())
        self.assertEqual(result['transferred_search_targets'], 0)

    def external(self, outcome='G'):
        value = dict(schema='paisho-external-outcome-v1', record_sha256=digest(self.input.read_bytes()),
                     outcome=outcome, kind='site_resignation', source_records=[dict(
                         game_id=42, original_sha256='a'*64, metadata_sha256='b'*64)])
        self.input.with_suffix('.outcome.json').write_text(json.dumps(value))
        return value

    def test_resignation_is_rebound_when_revalidated_game_is_still_ongoing(self):
        self.input.write_text('\n'.join(FIXTURE.splitlines()[:8])+'\n')
        original = self.external()
        result = self.run_migration()
        self.assertEqual(result['converted'], 1)
        target = self.record()
        sidecar = json.loads(target.with_suffix('.outcome.json').read_text())
        self.assertEqual(sidecar['record_sha256'], digest(target.read_bytes()))
        self.assertNotEqual(sidecar['record_sha256'], original['record_sha256'])
        self.assertEqual(sidecar['source_records'], original['source_records'])
        self.assertEqual(json.loads(self.input.with_suffix('.outcome.json').read_text()), original)

    def test_earlier_terminal_supersedes_later_resignation(self):
        self.input.write_text('\n'.join(FIXTURE.splitlines()[:-1])+'\n')
        self.external('H')
        result = self.run_migration()
        self.assertEqual(result['outcomes'], {'G': 1})
        self.assertFalse(self.record().with_suffix('.outcome.json').exists())
        receipt = json.loads((self.output / 'records.json').read_text())[0]
        self.assertEqual(receipt['external_policy'], 'superseded_by_earlier_rules_terminal')
        self.assertEqual(receipt['original_external_outcome']['outcome'], 'H')

    def test_conflicting_source_hash_is_excluded(self):
        self.input.write_text('\n'.join(FIXTURE.splitlines()[:8])+'\n')
        invalid = self.external()
        invalid['record_sha256'] = '0'*64
        self.input.with_suffix('.outcome.json').write_text(json.dumps(invalid))
        result = self.run_migration()
        self.assertEqual(result['converted'], 0)
        self.assertEqual(result['excluded'], 1)
        self.assertEqual(list((self.output / 'records').iterdir()), [])

    def test_duplicate_without_result_can_share_known_resignation(self):
        self.input.write_text('\n'.join(FIXTURE.splitlines()[:8])+'\n')
        self.external('H')
        (self.source / 'duplicate.psr').write_bytes(self.input.read_bytes())
        result = self.run_migration()
        self.assertEqual(result['converted'], 2)
        self.assertEqual(result['unique_records'], 1)
        for target in (self.output / 'records').glob('*.psr'):
            sidecar = json.loads(target.with_suffix('.outcome.json').read_text())
            self.assertEqual(sidecar['outcome'], 'H')

    def test_duplicate_with_opposite_known_result_is_excluded(self):
        self.input.write_text('\n'.join(FIXTURE.splitlines()[:8])+'\n')
        first = self.external('H')
        duplicate = self.source / 'duplicate.psr'
        duplicate.write_bytes(self.input.read_bytes())
        duplicate.with_suffix('.outcome.json').write_text(json.dumps(dict(first, outcome='G')))
        result = self.run_migration()
        self.assertEqual(result['converted'], 0)
        self.assertEqual(result['excluded'], 2)

    def test_repeated_migration_keeps_original_split(self):
        self.run_migration()
        target = self.record()
        first = json.loads(target.with_suffix('.rules-migration.json').read_text())
        next_output = self.root / 'second'
        migrate(self.output / 'records', next_output, CONVERTER)
        next_record = next((next_output / 'records').glob('*.psr'))
        second = json.loads(next_record.with_suffix('.rules-migration.json').read_text())
        self.assertEqual(first['split_identity_sha256'], second['split_identity_sha256'])

    def test_existing_directory_is_never_reused(self):
        self.run_migration()
        before = self.record().read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_migration()
        self.assertEqual(self.record().read_bytes(), before)

    def test_distinct_sources_collapsing_to_same_prefix_do_not_cross_splits(self):
        other = self.source / 'other.psr'
        other.write_text('\n'.join(FIXTURE.splitlines()[:-1])+'\n')
        result = self.run_migration()
        self.assertEqual(result['excluded'], 2)
        self.assertEqual(result['converted'], 0)

    def test_invalid_action_before_terminal_is_excluded(self):
        self.input.write_text(FIXTURE.replace('arrange 0,-8 2,-5', 'arrange 0,-8 0,8'))
        result = self.run_migration()
        self.assertEqual(result['excluded'], 1)
        self.assertEqual(result['converted'], 0)


if __name__ == '__main__':
    unittest.main()
