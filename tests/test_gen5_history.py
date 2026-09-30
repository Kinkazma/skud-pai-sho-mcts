import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('history',Path(__file__).resolve().parents[1]/'experimental/gen5/tools/continue_history.py')
history=importlib.util.module_from_spec(spec);spec.loader.exec_module(history)


class HistoryPortability(unittest.TestCase):
    def test_relocation_does_not_round_numeric_tokens_or_source_ids(self):
        raw=b'{"path":"assets/gen5-history/archives/old/x","weight":1.234567890123456789,"source":"portable-source/123","count":18446744073709551615}'
        moved=history.rewrite_strings(raw,[('assets/gen5-history/archives','runs/example/archives')])
        self.assertEqual(moved,raw.replace(b'assets/gen5-history/archives',b'runs/example/archives'))

    def test_atomic_revision_update_isolated_from_installed_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            (source/'revisions').mkdir();original=source/'revisions/lesson.json.gz';original.write_bytes(b'original')
            target=root/'working';self.assertEqual(history.link_archive(source,target),1)
            self.assertEqual(original.stat().st_ino,(target/'revisions/lesson.json.gz').stat().st_ino)
            # The native writer creates .tmp then renames; it never truncates originals.
            tmp=target/'revisions/lesson.tmp';tmp.write_bytes(b'new');tmp.replace(target/'revisions/lesson.json')
            self.assertEqual(original.read_bytes(),b'original')
            self.assertFalse((source/'revisions/lesson.json').exists())
            tmp.write_bytes(b'replaced');tmp.replace(target/'revisions/lesson.json.gz')
            self.assertEqual(original.read_bytes(),b'original')

    def test_existing_run_is_never_reinitialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'runs/existing';out.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'already exists'):
                history.prepare(root,out,'prepared',10)


if __name__=='__main__':unittest.main()
