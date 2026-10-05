"""A separately trained non-check stand-pat adapter; old quiet models stay gated."""
import chess
import numpy as np
import torch
from engine.app_evaluation import evaluate as app_evaluate
from engine.strategic import PRIORS, StrategicEvaluator
from ml.incremental import IncrementalEncoder
from ml.model import RELATIONAL_INPUT_SIZE, SCORE_SCALE
from ml.threat_model import (THREAT_INPUT_SIZE, THREAT_VERSION, ThreatNet,
                            mirror_array)
from ml.standpat_features import threat_features_with_attacks

GATE_ID = 'noncheck_standpat_v2'


def gate_active(board):
    return not board.is_check()


class StandPatHybrid:
    cacheable_by_fen = True

    def __init__(self, checkpoint, weight=.25):
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (saved.get('version') != THREAT_VERSION or saved.get('input_size') != THREAT_INPUT_SIZE
                or saved.get('score_scale') != SCORE_SCALE
                or saved.get('baseline_id') != 'strategic-v26' or saved.get('baseline_weights') != list(PRIORS)
                or saved.get('training_correction_weight') != weight
                or saved.get('training_quiet_only') is not False
                or saved.get('training_gate') != GATE_ID
                or saved.get('training_target') != 'stockfish_static_white_cp'):
            raise ValueError('Requires a matching static-trained stand-pat checkpoint')
        self.weight = weight; self.model = ThreatNet()
        self.model.load_state_dict(saved['state_dict']); self.model.eval()
        self.baseline = StrategicEvaluator(PRIORS); self.encoder = IncrementalEncoder(RELATIONAL_INPUT_SIZE)
        layers = [self.model.net[i] for i in (0,2,4)]
        self.weights = [l.weight.detach().numpy() for l in layers]
        self.biases = [l.bias.detach().numpy() for l in layers]
        self.buffers = [np.empty(n,dtype=np.float32) for n in (64,32,1)]

    def evaluate_position(self, board):
        base = self.baseline.evaluate_position(board)
        if not gate_active(board):
            return base
        threats,attacks = threat_features_with_attacks(board)
        x = np.concatenate((self.encoder.encode(board,attacks),threats))
        sign = 1 if board.turn else -1
        if sign == -1:
            x = mirror_array(x)
        for i,(w,bias,buffer) in enumerate(zip(self.weights,self.biases,self.buffers)):
            np.matmul(w,x,out=buffer); np.add(buffer,bias,out=buffer)
            if i < 2:
                np.maximum(buffer,0,out=buffer)
            x = buffer
        correction = 250*np.tanh(float(x[0])/(250/SCORE_SCALE))
        return round(base + sign*self.weight*correction)

    def __call__(self, board):
        return app_evaluate(board) if board.is_game_over(claim_draw=False) else self.evaluate_position(board)
