import unittest
import xml.etree.ElementTree as ET

from analyze_metal_shaders import summarize


class ShaderSamplesTests(unittest.TestCase):
    def test_references_process_filter_boundary_and_overlap(self):
        xml = '''<trace-query-result><node><schema name="metal-shader-profiler-intervals">
          <col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col>
          <col><mnemonic>name</mnemonic></col><col><mnemonic>process</mnemonic></col></schema>
          <row><t>5000000000</t><d id="1">2000000000</d><s id="2">conv (17)</s>
          <p id="3"><pid>42</pid></p></row>
          <row><t>6000000000</t><d ref="1"/><s ref="2"/><p ref="3"/></row>
          <row><t>6000000000</t><d ref="1"/><s>other</s><p><pid>7</pid></p></row>
          </node></trace-query-result>'''
        result = summarize(ET.fromstring(xml), 42, 6, 8)
        self.assertEqual(result["summed_sample_seconds"], 3)
        self.assertEqual(result["shaders"], [{"name": "conv", "samples": 2,
            "summed_sample_seconds": 3, "share_of_summed_sample_duration_percent": 100}])
        self.assertEqual(summarize(ET.fromstring(xml), 99, 6, 8)["shaders"], [])
        with self.assertRaises(ValueError):
            summarize(ET.fromstring(xml), 42, 8, 6)


if __name__ == "__main__":
    unittest.main()
