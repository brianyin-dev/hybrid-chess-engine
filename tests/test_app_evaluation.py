import unittest
import chess
from engine.app_evaluation import evaluate, search_evaluator
from engine.evaluation import evaluate as original
from engine.strategic import PRIORS, StrategicEvaluator


class AppEvaluationTests(unittest.TestCase):
    def test_uses_selected_strategic_weights(self):
        board=chess.Board('7k/8/P7/8/8/8/8/7K w - - 0 1')
        self.assertEqual(evaluate(board),StrategicEvaluator(PRIORS)(board))
        self.assertNotEqual(evaluate(board),original(board))
        self.assertIsNot(search_evaluator(),search_evaluator())

    def test_terminal_rules_preserved(self):
        for fen in ('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1',
                    '7k/5Q2/5K2/8/8/8/8/8 b - - 0 1',
                    '7k/8/8/8/8/8/8/7K w - - 0 1'):
            board=chess.Board(fen)
            self.assertEqual(evaluate(board),original(board))
