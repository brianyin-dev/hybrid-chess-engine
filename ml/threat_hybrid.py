"""Experimental relationship evaluator pinned to strategic-v26 and its quiet gate."""
import chess
import numpy as np
import torch
from engine.app_evaluation import evaluate as app_evaluate
from engine.strategic import PRIORS, StrategicEvaluator
from ml.incremental import IncrementalEncoder
from ml.model import RELATIONAL_INPUT_SIZE, SCORE_SCALE
from ml.threat_model import (THREAT_INPUT_SIZE, THREAT_VERSION, ThreatNet,
                            threat_features, mirror_array)


class ThreatHybrid:
    cacheable_by_fen = True

    def __init__(self, checkpoint, weight=.25):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (saved.get('version') != THREAT_VERSION or saved.get('input_size') != THREAT_INPUT_SIZE
                or saved.get('baseline_id') != 'strategic-v26'
                or saved.get('baseline_weights') != list(PRIORS)
                or saved.get('training_correction_weight') != weight
                or saved.get('score_scale') != SCORE_SCALE
                or not saved.get('training_quiet_only')):
            raise ValueError('incompatible threat checkpoint or runtime blend')
        self.weight = weight
        self.model = ThreatNet()
        self.model.load_state_dict(saved['state_dict'])
        self.model.eval()
        self.baseline = StrategicEvaluator(PRIORS)
        self.encoder = IncrementalEncoder(RELATIONAL_INPUT_SIZE)
        layers = [self.model.net[i] for i in (0, 2, 4)]
        self.weights = [l.weight.detach().numpy() for l in layers]
        self.biases = [l.bias.detach().numpy() for l in layers]

    def evaluate_position(self, board):
        base = self.baseline.evaluate_position(board)
        if board.is_check() or next(board.generate_legal_captures(), None) is not None:
            return base
        x = np.concatenate((self.encoder.encode(board), threat_features(board)))
        sign = 1 if board.turn else -1
        if sign == -1:
            x = mirror_array(x)
        for w, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(w @ x + bias, 0)
        raw = float((self.weights[-1] @ x + self.biases[-1])[0])
        correction = 250 * np.tanh(raw / (250 / SCORE_SCALE))
        return round(base + sign * self.weight * correction)

    def __call__(self, board):
        return app_evaluate(board) if board.is_game_over(claim_draw=False) else self.evaluate_position(board)
