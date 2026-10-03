"""Jointly tunable positional relationships with a fixed material/PST baseline.

All features are symmetric White-minus-Black scores. No position-specific rules,
terminal-rule guesses or special treatment of a previously observed move.
"""
import chess,math
from engine import evaluation as hce

FEATURE_NAMES=('safe_mobility','king_danger','piece_pressure','passed_pawn_danger','pawn_structure','bishop_pair')
PRIORS=(1.,1.,.25,1.,1.,1.)
BOUNDS=(2.,2.,1.,2.,2.,2.)

class StrategicEvaluator:
    cacheable_by_fen=True
    position_only=True
    def __init__(self,weights=None,baseline=hce):
        self.weights=tuple(weights or (0.,)*6)
        if len(self.weights)!=6:raise ValueError('six strategic weights required')
        if any(not math.isfinite(w) or not 0<=w<=limit for w,limit in zip(self.weights,BOUNDS)):
            raise ValueError('strategic weights must be finite and within bounds')
        self.hce=baseline
        self.pieces={};self.activities={}
        self.tables={}
        for color in chess.COLORS:
            for pt,table in baseline.PST.items():
                oriented=tuple(table[sq^56 if color else sq] for sq in range(64))
                self.tables[color,pt]=oriented
                if pt==chess.KING:
                    for phase in range(25):
                        self.tables[color,pt,phase]=tuple((oriented[sq]*phase+(40-10*(abs(2*(sq&7)-7)+abs(2*(sq>>3)-7)))*(24-phase))//24 for sq in range(64))

    def analyze(self,b,features=True):
        e=self.hce;phase=e._phase(b)
        attacks={s:b.attacks_mask(s) for s in chess.scan_forward(b.occupied)}
        combined={c:0 for c in chess.COLORS};pawn_attacks={c:0 for c in chess.COLORS}
        for c in chess.COLORS:
            for s in chess.scan_forward(b.occupied_co[c]):combined[c]|=attacks[s]
            if features:
                for s in b.pieces(chess.PAWN,c):pawn_attacks[c]|=attacks[s]
        base=0;result=[0.]*6
        for c in chess.COLORS:
            sign=1 if c else -1;enemy=not c;own=b.occupied_co[c]
            score=0
            for pt in chess.PIECE_TYPES:
                mask=b.pieces_mask(pt,c);p=phase if pt==chess.KING else -1;key=(mask,p)
                prior=self.pieces.get((c,pt))
                if prior is None or prior[0]!=key:
                    table=self.tables[c,pt,phase] if pt==chess.KING else self.tables[c,pt]
                    prior=(key,mask.bit_count()*e.MATERIAL[pt]+sum(table[s] for s in chess.scan_forward(mask)))
                    self.pieces[c,pt]=prior
                score+=prior[1]
            mobility=0;safe_mobility=0
            for pt,w in e.MOBILITY_WEIGHTS.items():
                for s in b.pieces(pt,c):
                    destinations=attacks[s]&~own
                    mobility+=w*destinations.bit_count()
                    if features:safe_mobility+=w*(destinations&~pawn_attacks[enemy]).bit_count()
            score+=mobility+e.CENTER_CONTROL_BONUS*sum(bool(combined[c]&chess.BB_SQUARES[s]) for s in e.CENTER_SQUARES)
            for name,key,fn in (
                ('development',(b.knights&own,b.bishops&own,b.queens&own,phase),lambda:e._development_score(b,c,phase)),
                ('rook',(b.rooks&own,b.pawns&own,b.pawns&b.occupied_co[enemy]),lambda:e._rook_activity_score(b,c)),
                ('passed',(b.pawns&own,b.pawns&b.occupied_co[enemy]),lambda:e._passed_pawn_score(b,c))):
                prior=self.activities.get((c,name))
                if prior is None or prior[0]!=key:prior=(key,fn());self.activities[c,name]=prior
                score+=prior[1]
            base+=sign*score
            if not features:continue
            result[0]+=sign*(safe_mobility-mobility)
            king=b.king(c);enemy_king=b.king(enemy)
            if king is not None:
                ring=chess.BB_KING_ATTACKS[king]|chess.BB_SQUARES[king];units=attackers=0
                for pt,w in ((chess.PAWN,1),(chess.KNIGHT,2),(chess.BISHOP,2),(chess.ROOK,3),(chess.QUEEN,4)):
                    for s in b.pieces(pt,enemy):
                        hits=(attacks[s]&ring).bit_count()
                        if hits:units+=w*hits;attackers+=1
                # Nonlinear cooperation, reduced without an enemy queen and in endings.
                danger=min(500.,units*units*min(attackers,4)/8.)
                direction=1 if c else -1;shield=0
                for file in range(max(0,(king&7)-1),min(8,(king&7)+2)):
                    for distance in (1,2):
                        rank=(king>>3)+direction*distance
                        if 0<=rank<8 and b.pieces_mask(chess.PAWN,c)&chess.BB_SQUARES[chess.square(file,rank)]:shield+=1;break
                danger+=(3-shield)*8 if b.queens&b.occupied_co[enemy] else 0
                result[1]-=sign*danger*(phase/24)*(1. if b.queens&b.occupied_co[enemy] else .35)
            pressure=0.
            for pt in (chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN):
                for s in b.pieces(pt,c):
                    if not combined[enemy]&chess.BB_SQUARES[s]:continue
                    cheapest=min((20000 if b.piece_type_at(a)==chess.KING else e.MATERIAL[b.piece_type_at(a)]) for a in b.attackers(enemy,s))
                    loss=max(0,e.MATERIAL[pt]-cheapest) if combined[c]&chess.BB_SQUARES[s] else e.MATERIAL[pt]
                    pressure+=loss
            result[2]-=sign*min(600.,pressure)/4
            pawns=b.pieces_mask(chess.PAWN,c);files={s&7 for s in chess.scan_forward(pawns)}
            structure=12*sum((s&7)-1 not in files and (s&7)+1 not in files for s in chess.scan_forward(pawns))
            structure+=10*(pawns.bit_count()-len(files));result[4]-=sign*structure
            result[5]+=sign*(30 if (b.bishops&own).bit_count()>=2 else 0)
            for s in chess.scan_forward(pawns):
                if b.pawns&b.occupied_co[enemy]&e.PASSED_MASKS[c][s]:continue
                progress=(s>>3) if c else 7-(s>>3)
                if progress<4:continue
                front=s+(8 if c else -8)
                if not 0<=front<64 or b.occupied&chess.BB_SQUARES[front]:continue
                bonus=(progress-3)**2*10
                if combined[c]&chess.BB_SQUARES[front]:bonus*=1.25
                if combined[enemy]&chess.BB_SQUARES[front]:bonus*=.4
                promotion=chess.square(s&7,7 if c else 0)
                if enemy_king is not None:bonus+=max(-30,min(30,(chess.square_distance(enemy_king,promotion)-(7-progress))*10))
                # Square rule only when the opponent has no other pieces and the route is empty.
                route=chess.BB_FILES[s&7]&e.PASSED_MASKS[c][s]
                other=b.occupied_co[enemy]&~(b.pawns|b.kings)
                # An opponent moving first has MORE time to catch the pawn.
                # Include that tempo in the king's catch radius, not subtract it.
                moves=7-progress+(1 if b.turn!=c else 0)
                if not other and not b.occupied&route and enemy_king is not None and chess.square_distance(enemy_king,promotion)>moves:bonus+=250
                result[3]+=sign*min(400,bonus)*(1-phase/32)
        return base,result

    def evaluate_position(self,b):
        base,features=self.analyze(b,any(self.weights))
        return round(base+sum(w*x for w,x in zip(self.weights,features)))
    __call__=evaluate_position
