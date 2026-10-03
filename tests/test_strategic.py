import math,random,unittest
import chess
from engine.strategic import StrategicEvaluator,PRIORS
from benchmarks.evaluation_baseline_v26 import evaluate_position
from engine.search import _Search

class StrategicEvaluationTests(unittest.TestCase):
    def test_opponent_first_move_cannot_improve_promotion_bonus(self):
        e=StrategicEvaluator(PRIORS)
        for king in ('d5','e5'):
            b=chess.Board(None);b.set_piece_at(chess.H1,chess.Piece(chess.KING,True))
            b.set_piece_at(chess.A5,chess.Piece(chess.PAWN,True));b.set_piece_at(chess.parse_square(king),chess.Piece(chess.KING,False))
            b.turn=True;_,first=e.analyze(b);b.turn=False;_,second=e.analyze(b)
            self.assertLessEqual(second[3],first[3])
            if king=='e5':self.assertLess(second[3],first[3])

    def test_position_only_cache_reuses_clocks_but_preserves_clock_aware_models(self):
        b=chess.Board();e=StrategicEvaluator();w=_Search(b,e,None,True)
        a=w.static(b);b.halfmove_clock=12
        self.assertEqual(w.static(b),a);self.assertEqual(w.static_cache_hits,1)
        class ClockModel:
            cacheable_by_fen=True
            def __call__(self,b):return b.halfmove_clock
        m=_Search(b,ClockModel(),None,True)
        self.assertEqual(m.static(b),12);b.halfmove_clock=13
        self.assertEqual(m.static(b),13);self.assertEqual(m.static_cache_hits,0)
        b.halfmove_clock=150
        self.assertEqual(w.terminal(b,next(iter(b.legal_moves)),0),0)

    def test_zero_weights_exactly_match_original_across_history(self):
        rng=random.Random(260028);b=chess.Board();e=StrategicEvaluator()
        for _ in range(100):
            self.assertEqual(e(b),evaluate_position(b))
            if b.is_game_over():b=chess.Board()
            b.push(rng.choice(list(b.legal_moves)))

    def test_general_features_are_color_symmetric(self):
        rng=random.Random(260029);b=chess.Board();e=StrategicEvaluator(PRIORS)
        for _ in range(60):
            before=b.fen();a,x=e.analyze(b);z,y=e.analyze(b.mirror())
            self.assertEqual(a,-z)
            for v,w in zip(x,y):self.assertAlmostEqual(v,-w)
            self.assertTrue(all(math.isfinite(v) for v in x));self.assertEqual(b.fen(),before)
            if b.is_game_over():b=chess.Board()
            b.push(rng.choice(list(b.legal_moves)))

    def test_material_values_remain_fixed(self):
        b=chess.Board('7k/8/8/8/8/8/4Q3/K7 w - - 0 1')
        e=StrategicEvaluator();a,_=e.analyze(b)
        self.assertEqual(a,evaluate_position(b))
        b.remove_piece_at(chess.E2)
        self.assertGreater(a-e(b),850)
