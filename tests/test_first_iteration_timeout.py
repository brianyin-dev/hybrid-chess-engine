"""Regression coverage for safe use of completed work before depth one finishes."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import chess
from engine.search import search, _Search, _SearchLimit
from benchmarks.pre_timeout.search import search as previous_search

class FirstIterationTimeoutTests(unittest.TestCase):
    def test_no_completed_root_keeps_emergency_fallback(self):
        board=chess.Board();before=board.fen(),list(board.move_stack)
        with patch.object(_Search,'quiescence',side_effect=_SearchLimit('node_limit')):
            r=search(board,depth=1)
        self.assertFalse(r.used_partial_root)
        self.assertEqual(r.depth,0);self.assertIsNone(r.score)
        self.assertIn(r.move,board.legal_moves)
        self.assertEqual((board.fen(),board.move_stack),before)

    def test_interrupted_pvs_verification_does_not_publish_probe(self):
        board=chess.Board();before=board.fen(),list(board.move_stack)
        fallback=chess.Move.from_uci('e2e4')
        # First alternative scores10. Second probes20 but never completes its
        # required full-window verification; it must not replace the first.
        with patch.object(_Search,'fallback',return_value=fallback),patch.object(
                _Search,'quiescence',side_effect=[-10,-20,_SearchLimit('time_limit')]):
            r=search(board,depth=1)
        self.assertEqual(r.move,fallback);self.assertTrue(r.used_partial_root)
        self.assertEqual(r.depth,0);self.assertIsNone(r.score)
        self.assertEqual((board.fen(),board.move_stack),before)

    def test_verified_improvement_is_retained_not_fail_low(self):
        board=chess.Board();fallback=chess.Move.from_uci('e2e4')
        moves=[fallback]+[m for m in board.legal_moves if m!=fallback]
        with patch.object(_Search,'fallback',return_value=fallback),patch.object(
                _Search,'quiescence',side_effect=[-10,-20,-20,0,_SearchLimit('time_limit')]):
            r=search(board,depth=1)
        self.assertEqual(r.move,moves[1]);self.assertTrue(r.used_partial_root)
        self.assertIsNone(r.score);self.assertEqual(r.depth,0)
        self.assertEqual(board.fen(),chess.STARTING_FEN)

    def test_later_interruption_keeps_last_completed_iteration(self):
        original=_Search.negamax
        def interrupt(worker,board,depth,alpha,beta,ply):
            if ply==0 and depth==2:
                worker.root_completed=(chess.Move.from_uci('a2a3'),999)
                raise _SearchLimit('time_limit')
            return original(worker,board,depth,alpha,beta,ply)
        expected=search(chess.Board(),depth=1)
        with patch.object(_Search,'negamax',interrupt):
            actual=search(chess.Board(),depth=3)
        self.assertEqual((actual.move,actual.score,actual.depth),
                         (expected.move,expected.score,1))
        self.assertFalse(actual.used_partial_root)

    def test_fixed_depth_results_unchanged(self):
        for fen in [chess.STARTING_FEN,
                    '3r3k/8/8/3p4/8/8/8/K2Q4 w - - 0 1',
                    '7k/8/8/3pP3/8/8/8/K7 w - d6 0 1']:
            old=previous_search(chess.Board(fen),depth=2)
            new=search(chess.Board(fen),depth=2)
            self.assertEqual((new.move,new.score,new.depth),(old.move,old.score,old.depth))
            self.assertFalse(new.used_partial_root)

    def test_recorded_neural_failures_with_deterministic_node_budget(self):
        try:
            from ml.relationship_hybrid import RelationshipHybrid
        except ImportError:
            self.skipTest('Optional neural dependencies unavailable')
        root=Path(__file__).resolve().parents[1]
        checkpoint=root/'ml/artifacts/relationship-v35/current-epoch-40.pt'
        if not checkpoint.exists():self.skipTest('Optional checkpoint unavailable')
        records=json.loads((root/'ml/artifacts/relationship-v35/user-pilot-20-250ms/report.json').read_text())['games']
        for game,ply,nodes,expected in [(11,10,850,'g4f3'),(15,20,1627,'d8f6')]:
            g=records[game-1];b=chess.Board(g['initial_fen'])
            for u in g['opening_moves']:b.push_uci(u)
            for m in g['moves'][:ply-1]:b.push_uci(m['uci'])
            before=b.fen(),list(b.move_stack)
            old=previous_search(b,depth=64,node_limit=nodes,eval_fn=RelationshipHybrid(checkpoint),use_lmr=True)
            new=search(b,depth=64,node_limit=nodes,eval_fn=RelationshipHybrid(checkpoint),use_lmr=True)
            self.assertEqual(old.move.uci(),g['moves'][ply-1]['uci'])
            self.assertEqual(new.move.uci(),expected)
            self.assertEqual(new.depth,0);self.assertIsNone(new.score)
            self.assertTrue(new.used_partial_root)
            self.assertEqual((b.fen(),b.move_stack),before)

if __name__=='__main__':unittest.main()
