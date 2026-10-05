"""Default app evaluator; original evaluation remains available for benchmarks."""
import chess
from engine.evaluation import evaluate as original_evaluate
from engine.strategic import PRIORS, StrategicEvaluator


def evaluate(board: chess.Board) -> int:
    if board.is_game_over(claim_draw=False):
        return original_evaluate(board)
    return StrategicEvaluator(PRIORS).evaluate_position(board)


def search_evaluator():
    """A separate mutable cache for each request/search."""
    return StrategicEvaluator(PRIORS)
