"""Residual NN adapter pinned to the strategic-v26 baseline."""
import chess
import torch
from engine.app_evaluation import evaluate as app_evaluate
from engine.strategic import PRIORS, StrategicEvaluator
from ml.evaluator import NeuralEvaluator


class StrategicHybrid:
    cacheable_by_fen = True

    def __init__(self, checkpoint, weight):
        metadata = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if metadata.get('baseline_id') != 'strategic-v26' or metadata.get('baseline_weights') != list(PRIORS):
            raise ValueError('checkpoint must be trained against strategic-v26')
        if metadata.get('training_correction_weight') != weight or not metadata.get('training_quiet_only'):
            raise ValueError('runtime blend must match training')
        self.neural = NeuralEvaluator(checkpoint,weight,True,incremental=True,fast_features=True)
        self.baseline = StrategicEvaluator(PRIORS)

    def evaluate_position(self, board):
        return self.neural._score(board,self.baseline.evaluate_position(board))

    def __call__(self, board):
        if board.is_game_over(claim_draw=False):
            return app_evaluate(board)
        return self.evaluate_position(board)
