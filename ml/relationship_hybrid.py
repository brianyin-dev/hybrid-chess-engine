"""Controlled v35 current-input and relational-input evaluator arms."""
import chess
import numpy as np
import torch
from torch import nn
from ml.model import ChessNet,SCORE_SCALE,RELATIONAL_INPUT_SIZE
from ml.threat_model import THREAT_INPUT_SIZE,board_to_threat_array,mirror_array
from ml.relationship_features import RELATIONSHIP_SIZE,KING_FEATURES,PAWN_FEATURES,relationship_features,mirror_relationship
from ml.incremental import IncrementalEncoder
from ml.standpat_features import threat_features_with_attacks
from ml.standpat_hybrid import GATE_ID
from engine.strategic import StrategicEvaluator,PRIORS
from engine.app_evaluation import evaluate as app_evaluate
ARMS=('current','relationships');VERSION=7;TARGET='fresh_confirmed_search_rankings_v35'

def features(board,arm):
    if arm not in ARMS:raise ValueError('Unknown representation')
    base=board_to_threat_array(board)
    return base if arm=='current' else np.concatenate((base,relationship_features(board)))

def mirror_features(x,arm):
    base=mirror_array(x[:THREAT_INPUT_SIZE])
    return base if arm=='current' else np.concatenate((base,mirror_relationship(x[THREAT_INPUT_SIZE:])))

class RelationshipNet(ChessNet):
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError('Unknown representation')
        self.arm=arm;size=THREAT_INPUT_SIZE+(RELATIONSHIP_SIZE if arm=='relationships' else 0)
        super().__init__(size,250,False,(64,32))
        # Exact signed affine permutation for Board.mirror, including turn/material scalars.
        idx=np.arange(size,dtype=np.float32);permutation=mirror_features(idx,arm).astype(np.int64)
        permutation[768]=768;permutation[792]=792
        sign=np.ones(size,dtype=np.float32);sign[768]=sign[792]=-1
        offset=np.zeros(size,dtype=np.float32);offset[768]=1
        self.register_buffer('mirror_index',torch.from_numpy(permutation),persistent=False)
        self.register_buffer('mirror_sign',torch.from_numpy(sign),persistent=False)
        self.register_buffer('mirror_offset',torch.from_numpy(offset),persistent=False)
    def forward(self,x):
        white=x[...,768]>.5
        mirrored=x[...,self.mirror_index]*self.mirror_sign+self.mirror_offset
        canonical=torch.where(white.unsqueeze(-1),x,mirrored)
        return super().forward(canonical)*torch.where(white,1.,-1.)

def metadata(state,arm):
    return {'version':VERSION,'arm':arm,'input_size':THREAT_INPUT_SIZE+(RELATIONSHIP_SIZE if arm=='relationships' else 0),
            'score_scale':SCORE_SCALE,'baseline_id':'strategic-v26','baseline_weights':list(PRIORS),
            'training_correction_weight':.25,'training_gate':GATE_ID,'training_target':TARGET,
            'correction_limit_cp':250,'hidden_sizes':[64,32],'state_dict':state}

class RelationshipHybrid:
    cacheable_by_fen=True
    def __init__(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=True);arm=saved.get('arm')
        if arm not in ARMS or any(saved.get(k)!=v for k,v in metadata({},arm).items() if k!='state_dict'):
            raise ValueError('Requires matching v35 representation checkpoint')
        self.arm=arm;self.model=RelationshipNet(arm);self.model.load_state_dict(saved['state_dict']);self.model.eval()
        self.baseline=StrategicEvaluator(PRIORS);self.encoder=IncrementalEncoder(RELATIONAL_INPUT_SIZE)
        self.weights=[self.model.net[i].weight.detach().numpy() for i in (0,2,4)]
        self.biases=[self.model.net[i].bias.detach().numpy() for i in (0,2,4)]
        self.buffers=[np.empty(n,dtype=np.float32) for n in (64,32,1)]
    def evaluate_position(self,board):
        base=self.baseline.evaluate_position(board)
        if board.is_check():return base
        threat,attacks=threat_features_with_attacks(board)
        x=np.concatenate((self.encoder.encode(board,attacks),threat))
        if self.arm=='relationships':x=np.concatenate((x,relationship_features(board)))
        sign=1 if board.turn else -1
        if sign<0:x=mirror_features(x,self.arm)
        for i,(w,b,buf) in enumerate(zip(self.weights,self.biases,self.buffers)):
            np.matmul(w,x,out=buf);np.add(buf,b,out=buf)
            if i<2:np.maximum(buf,0,out=buf)
            x=buf
        correction=250*np.tanh(float(x[0])*SCORE_SCALE/250)
        return round(base+sign*.25*correction)
    def __call__(self,board):
        return app_evaluate(board) if board.is_game_over(claim_draw=False) else self.evaluate_position(board)
