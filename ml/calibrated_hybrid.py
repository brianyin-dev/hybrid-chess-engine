"""Experimental frozen-feature residual head with a 25cp final correction bound."""
import chess
import numpy as np
import torch
from ml.threat_model import ThreatNet,THREAT_VERSION,THREAT_INPUT_SIZE,mirror_array
from ml.model import SCORE_SCALE,RELATIONAL_INPUT_SIZE
from ml.incremental import IncrementalEncoder
from ml.standpat_features import threat_features_with_attacks
from ml.standpat_hybrid import GATE_ID
from engine.strategic import StrategicEvaluator,PRIORS
from engine.app_evaluation import evaluate as app_evaluate
TARGET='frozen_feature_retention_calibration_v34'
FINAL_BOUND_CP=25

class CalibratedNet(ThreatNet):
    def __init__(self):
        super().__init__()
        self.correction_limit_cp=100
        for parameter in self.net[:4].parameters():parameter.requires_grad_(False)

def metadata(state):
    return {'version':THREAT_VERSION,'input_size':THREAT_INPUT_SIZE,'score_scale':SCORE_SCALE,
            'baseline_id':'strategic-v26','baseline_weights':list(PRIORS),'training_correction_weight':.25,
            'training_gate':GATE_ID,'training_target':TARGET,'correction_limit_cp':100,
            'final_correction_bound_cp':FINAL_BOUND_CP,'hidden_sizes':[64,32],
            'frozen_hidden_layers':True,'state_dict':state}

class CalibratedHybrid:
    cacheable_by_fen=True
    def __init__(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        required={k:v for k,v in metadata({}).items() if k!='state_dict'}
        if any(saved.get(k)!=v for k,v in required.items()):
            raise ValueError('Requires matching frozen-feature v34 checkpoint')
        self.model=CalibratedNet();self.model.load_state_dict(saved['state_dict']);self.model.eval()
        self.baseline=StrategicEvaluator(PRIORS);self.encoder=IncrementalEncoder(RELATIONAL_INPUT_SIZE)
        self.weights=[self.model.net[i].weight.detach().numpy() for i in (0,2,4)]
        self.biases=[self.model.net[i].bias.detach().numpy() for i in (0,2,4)]
        self.buffers=[np.empty(n,dtype=np.float32) for n in (64,32,1)]
    def evaluate_position(self,board):
        base=self.baseline.evaluate_position(board)
        if board.is_check():return base
        threats,attacks=threat_features_with_attacks(board)
        x=np.concatenate((self.encoder.encode(board,attacks),threats));sign=1 if board.turn else -1
        if sign<0:x=mirror_array(x)
        for i,(w,b,buf) in enumerate(zip(self.weights,self.biases,self.buffers)):
            np.matmul(w,x,out=buf);np.add(buf,b,out=buf)
            if i<2:np.maximum(buf,0,out=buf)
            x=buf
        delta=FINAL_BOUND_CP*np.tanh(float(x[0])*SCORE_SCALE/100)
        return round(base+sign*delta)
    def __call__(self,board):
        return app_evaluate(board) if board.is_game_over(claim_draw=False) else self.evaluate_position(board)
