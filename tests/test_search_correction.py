import unittest
from ml.run_search_correction_v29 import family, interval


class SearchCorrectionTests(unittest.TestCase):
    def test_transposed_openings_share_split_family(self):
        a=['g1f3','d7d5','d2d4','g8f6','c2c4','e7e6']
        b=['d2d4','d7d5','g1f3','g8f6','c2c4','e7e6']
        self.assertEqual(family(a),family(b))

    def test_large_material_gap_cannot_be_reversed_by_small_correction(self):
        good={'fen':'7k/8/8/8/8/8/4R3/K7 w - - 0 1'}
        bad={'fen':'7k/8/8/8/8/8/4Q3/K7 w - - 0 1'}
        for weight in (.1,.25):
            result=interval(good,bad,1,weight)
            self.assertLess(result['maximum_signed_margin_cp'],0)
            self.assertFalse(result['permits_correct_ranking'])

    def test_capture_gate_removes_endpoint_correction_capacity(self):
        leaf={'fen':'7k/8/8/8/8/3p4/2P5/K7 w - - 0 1'}
        result=interval(leaf,leaf,1,.25)
        self.assertEqual(result['maximum_signed_margin_cp'],0)
        self.assertFalse(result['permits_correct_ranking'])
