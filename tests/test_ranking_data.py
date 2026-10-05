import unittest
import chess
from ml.run_ranking_v30 import pair_key


class RankingDataTests(unittest.TestCase):
    def test_color_mirrors_are_same_ranking_example(self):
        good=chess.Board();good.push_uci('e2e4')
        bad=chess.Board();bad.push_uci('d2d4')
        self.assertEqual(pair_key(good.fen(),bad.fen()),
                         pair_key(good.mirror().fen(),bad.mirror().fen()))

    def test_reversing_teacher_preference_is_not_same_example(self):
        good=chess.Board();good.push_uci('e2e4')
        bad=chess.Board();bad.push_uci('d2d4')
        self.assertNotEqual(pair_key(good.fen(),bad.fen()),pair_key(bad.fen(),good.fen()))
