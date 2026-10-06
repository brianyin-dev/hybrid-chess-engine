"""Color-relative, location-sharing relationships; descriptive features, not SEE."""
import chess
import numpy as np
from engine.evaluation import MATERIAL,PASSED_MASKS
from ml.threat_model import pin_aware_attackers

KING_FEATURES=288  # 2 colors * 2 king relationships * 6 types * (8 distances + 4 directions)
PAWN_FEATURES=48   # 2 colors * 8 promotion distances * (count, passed, blocked)
VALUE_FEATURES=60  # 2 colors * 5 victim types * 6 attack/defense summaries
RELATIONSHIP_SIZE=KING_FEATURES+PAWN_FEATURES+VALUE_FEATURES

def relationship_features(board):
    kings=np.zeros((2,2,6,12),dtype=np.float32)
    pawns=np.zeros((2,8,3),dtype=np.float32)
    values=np.zeros((2,5,6),dtype=np.float32)
    for ci,color in enumerate((chess.WHITE,chess.BLACK)):
        direction=1 if color else -1
        for square in chess.scan_forward(board.occupied_co[color]):
            pt=board.piece_type_at(square);rank=chess.square_rank(square);file=chess.square_file(square)
            for ki,kcolor in enumerate((color,not color)):
                king=board.king(kcolor)
                if king is None:continue
                dx=file-chess.square_file(king);dy=direction*(rank-chess.square_rank(king))
                distance=max(abs(dx),abs(dy));quadrant=2*int(dy>=0)+int(dx>=0)
                kings[ci,ki,pt-1,distance]+=1/8
                kings[ci,ki,pt-1,8+quadrant]+=1/8
            if pt==chess.PAWN:
                distance=7-rank if color else rank
                forward=square+8*direction
                pawns[ci,distance,0]+=1/8
                pawns[ci,distance,1]+=float(not (board.pieces_mask(chess.PAWN,not color)&PASSED_MASKS[color][square]))/8
                pawns[ci,distance,2]+=float(0<=forward<64 and board.piece_type_at(forward) is not None)/8
            if pt==chess.KING:continue
            attackers=pin_aware_attackers(board,not color,square)
            if not attackers:continue
            defenders=pin_aware_attackers(board,color,square)
            victim=MATERIAL[pt]/900
            # Kings are expensive capture participants, not zero-value cheap attackers.
            cheapest=min(20000 if board.piece_type_at(a)==chess.KING else MATERIAL[board.piece_type_at(a)] for a in chess.scan_forward(attackers))
            defender=min((20000 if board.piece_type_at(a)==chess.KING else MATERIAL[board.piece_type_at(a)] for a in chess.scan_forward(defenders)),default=0)
            values[ci,pt-1,0]+=1/8
            values[ci,pt-1,1]+=victim/8
            values[ci,pt-1,2]+=victim*float(not defenders)/8
            values[ci,pt-1,3]+=max(0,MATERIAL[pt]-cheapest)/900/8
            values[ci,pt-1,4]+=min(defender,900)/900/8
            values[ci,pt-1,5]+=victim*float(board.is_pinned(color,square))/8
    return np.concatenate((kings.reshape(-1),pawns.reshape(-1),values.reshape(-1)))

def mirror_relationship(values):
    if values.shape!=(RELATIONSHIP_SIZE,):raise ValueError('Wrong relationship feature size')
    return np.concatenate((values[:KING_FEATURES].reshape(2,-1)[::-1].reshape(-1),
                           values[KING_FEATURES:KING_FEATURES+PAWN_FEATURES].reshape(2,-1)[::-1].reshape(-1),
                           values[KING_FEATURES+PAWN_FEATURES:].reshape(2,-1)[::-1].reshape(-1)))
