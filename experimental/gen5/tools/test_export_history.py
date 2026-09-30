import gzip, hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('export_history',Path(__file__).with_name('export_history.py'))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ExportTests(unittest.TestCase):
 def test_source_keys_and_numeric_tokens(self):
  raw=b'[{"source_run":"/Users/private/run","game_id":"42","value":-0.0000001234567890123456,"counter":18446744073709551615}]'
  out,n=mod.sanitize(raw);self.assertEqual(n,1)
  self.assertIn(b'"value":-0.0000001234567890123456',out)
  self.assertIn(b'18446744073709551615',out)
  expected=hashlib.sha256(b'/Users/private/run/42').hexdigest()
  self.assertEqual(json.loads(out)[0]['source_run'],'portable-source/'+expected)
  self.assertNotIn(b'/Users/',out)
 def test_archived_sort_key_and_valid_content_hash(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);raw=gzip.compress(b'{"source_run":"/Users/private/run","game_id":9,"value":0.25}',mtime=0)
   old=mod.sha(raw);src=root/(old+'.json.gz');src.write_bytes(raw)
   row=mod.export_one((src,root/'export'/src.name,'bundle',old))
   target=Path(row[1]);self.assertEqual(target.name,old+'__'+row[3]+'.json.gz')
   self.assertEqual(mod.sha(target.read_bytes()),row[3]);self.assertEqual(src.read_bytes(),raw)
   self.assertEqual(json.loads(gzip.decompress(target.read_bytes()))['value'],0.25)
 def test_corruption_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   src=Path(t)/('0'*64+'.json.gz');src.write_bytes(gzip.compress(b'{}'))
   with self.assertRaisesRegex(ValueError,'hash'):mod.export_one((src,Path(t)/'out.json.gz','bundle',None))
if __name__=='__main__':unittest.main()
