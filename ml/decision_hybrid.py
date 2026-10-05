"""Experimental v33 output shaping; v32 and production remain unchanged."""
import chess
import numpy as np
import torch
from ml.threat_model import ThreatNet,THREAT_INPUT_SIZE,THREAT_VERSION,mirror_array
from ml.model import SCORE_SCALE,RELATIONAL_INPUT_SIZE
from ml.incremental import IncrementalEncoder
from ml.standpat_features import threat_features_with_attacks
from ml.standpat_hybrid import GATE_ID
from engine.strategic import StrategicEvaluator,PRIORS
from engine.app_evaluation import evaluate as app_evaluate

OUTPUTS=('tanh','linear_clip')
TARGET='stockfish_confirmed_root_and_qleaf_rankings_v33'

class DecisionNet(ThreatNet):
    def __init__(self,output_mode='linear_clip'):
        if output_mode not in OUTPUTS:raise ValueError('Unknown output rule')
        super().__init__();self.correction_limit_cp=None;self.output_mode=output_mode
    def raw(self,x):
        return super().forward(x)
    def forward(self,x):
        raw=self.raw(x);limit=250/SCORE_SCALE
        return limit*torch.tanh(raw/limit) if self.output_mode=='tanh' else raw.clamp(-limit,limit)

class DecisionHybrid:
    cacheable_by_fen=True
    def __init__(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        required={'version':THREAT_VERSION,'input_size':THREAT_INPUT_SIZE,'score_scale':SCORE_SCALE,
                  'baseline_id':'strategic-v26','baseline_weights':list(PRIORS),'training_correction_weight':.25,
                  'training_quiet_only':False,'training_gate':GATE_ID,'training_target':TARGET,
                  'correction_limit_cp':250,'hidden_sizes':[64,32]}
        if any(saved.get(k)!=v for k,v in required.items()) or saved.get('output_mode') not in OUTPUTS:
            raise ValueError('Requires exact v33 decision checkpoint metadata')
        self.output_mode=saved['output_mode'];self.model=DecisionNet(self.output_mode)
        self.model.load_state_dict(saved['state_dict']);self.model.eval()
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
        raw=float(x[0])*SCORE_SCALE
        cp=250*np.tanh(raw/250) if self.output_mode=='tanh' else max(-250,min(250,raw))
        return round(base+sign*.25*cp)
    def __call__(self,board):
        return app_evaluate(board) if board.is_game_over(claim_draw=False) else self.evaluate_position(board)

def metadata(state,mode):
    return {'version':THREAT_VERSION,'input_size':THREAT_INPUT_SIZE,'score_scale':SCORE_SCALE,
            'baseline_id':'strategic-v26','baseline_weights':list(PRIORS),'training_correction_weight':.25,
            'training_quiet_only':False,'training_gate':GATE_ID,'training_target':TARGET,
            'correction_limit_cp':250,'hidden_sizes':[64,32],'output_mode':mode,'state_dict':state}
