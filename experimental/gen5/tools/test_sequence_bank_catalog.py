import struct
import tempfile
import unittest
from pathlib import Path
from sequence_bank_catalog import motifs, catalog


class SequenceCatalogTests(unittest.TestCase):
    def bank(self, spatial):
        header=(b'PSSEQ002' if spatial else b'PSSEQ001')+struct.pack('<4I',2,1,2,1)
        rows=[]
        for i in range(2):
            row=struct.pack('<64h',*([100+i]*64))+bytes(128)
            row+=struct.pack('<QIIIbB',i+1,i,10,20,1-i,i)
            if spatial:row+=bytes(200)
            rows.append(row)
        return header+b''.join(rows)+bytes(256)+struct.pack('<3I',2,0,1)

    def test_spatial_layout_keeps_game_motif_usage_and_human_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'old.bin';b=Path(directory)/'spatial.bin'
            a.write_bytes(self.bank(False));b.write_bytes(self.bank(True))
            usage={(1,10):7,(2,10):11}
            self.assertEqual(motifs(a,usage),motifs(b,usage))
            self.assertEqual(motifs(b,usage)[-1],{0:7,1:11})
            sources=[{'sha256':'a'*64,'human':True},{'sha256':'b'*64,'human':False}]
            self.assertEqual(catalog(a,sources),catalog(b,sources))
            self.assertTrue(catalog(b,sources)[0].human)

    def test_truncated_spatial_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bad.bin';p.write_bytes(self.bank(True)[:24+477])
            with self.assertRaisesRegex(ValueError,'truncated'):
                motifs(p)


if __name__=='__main__':
    unittest.main()
