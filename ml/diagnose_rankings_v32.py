"""Descriptive audit of exposed v32 test failures; no training or selection."""
import json
from collections import Counter
from pathlib import Path
import chess
import chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from ml.standpat_hybrid import StandPatHybrid

ART=Path('ml/artifacts/standpat-v32')
VALUES={chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}

def details(board):
    material=sum(VALUES[p.piece_type]*(1 if p.color else -1) for p in board.piece_map().values())
    exposed=[];pinned=[];advanced=[]
    for sq,p in board.piece_map().items():
        if p.piece_type!=chess.KING and board.attackers(not p.color,sq) and not board.attackers(p.color,sq):
            exposed.append(p.symbol()+'@'+chess.square_name(sq))
        if board.is_pinned(p.color,sq):pinned.append(p.symbol()+'@'+chess.square_name(sq))
        if p.piece_type==chess.PAWN and (chess.square_rank(sq)>=5 if p.color else chess.square_rank(sq)<=2):
            advanced.append(p.symbol()+'@'+chess.square_name(sq))
    return {'white_material_cp':material,'attacked_undefended':exposed,'pinned':pinned,'advanced_pawns':advanced,
            'has_captures':any(board.generate_legal_captures())}

def main():
    with exclusive_cpu('v32 exposed ranking failure diagnosis'):
        data=json.loads((ART/'collection.json').read_text());pairs=json.loads((ART/'pairs.json').read_text())['test']
        rows={r['fen']:r for r in data['rows']['test']};model=StandPatHybrid(ART/'epoch-8.pt')
        failures=[];counts=Counter();pred={}
        for p in pairs:
            for fen in (p['good_fen'],p['bad_fen']):
                if fen not in pred:pred[fen]=model.evaluate_position(chess.Board(fen))
            gap=p['sign']*(pred[p['good_fen']]-pred[p['bad_fen']])
            if gap>0:continue
            base=p['base_margin_cp'];delta=gap-base
            cause='wrong_direction' if delta<0 else 'no_effect' if delta==0 else 'right_direction_insufficient'
            counts[cause]+=1;counts[p['kind']]+=1;counts['ties']+=int(gap==0)
            counts['theoretically_unrepairable']+=int(p['maximum_signed_margin_cp']<=0)
            failures.append({**p,'hybrid_margin_cp':gap,'correction_margin_cp':delta,'cause':cause,
                             'good_static_cp':rows[p['good_fen']]['score_cp'], 'bad_static_cp':rows[p['bad_fen']]['score_cp'],
                             'good_details':details(chess.Board(p['good_fen'])), 'bad_details':details(chess.Board(p['bad_fen']))})
        engine=chess.engine.SimpleEngine.popen_uci('tools/stockfish-sf19/stockfish/stockfish-macos-universal')
        engine.configure({'Threads':1,'Hash':32})
        analyses={}
        try:
            for i,p in enumerate(failures):
                for key in ('good_fen','bad_fen'):
                    fen=p[key]
                    if fen in analyses:continue
                    r=rows[fen];b=chess.Board(r['initial_fen'])
                    for move in r['history']:b.push_uci(move)
                    info=engine.analyse(b,chess.engine.Limit(nodes=64000),game=object())
                    analyses[fen]={'white_cp':info['score'].white().score(),'white_mate':info['score'].white().mate(),
                                   'depth':info['depth'],'pv':[m.uci() for m in info.get('pv',[])[:10]]}
                a=analyses[p['good_fen']];b=analyses[p['bad_fen']]
                p['good_search']=a;p['bad_search']=b
                if a['white_cp'] is not None and b['white_cp'] is not None:
                    sg=p['sign']*(a['white_cp']-b['white_cp']);p['searched_teacher_margin_cp']=sg
                    counts['search_agrees']+=int(sg>0);counts['search_ties']+=int(sg==0);counts['search_reverses']+=int(sg<0)
                else:counts['search_mate_pairs']+=1
                if (i+1)%20==0:print(f'Checked {i+1}/{len(failures)} failures',flush=True)
        finally:engine.quit()
        out={'purpose':'Post-test diagnosis only; static and searched targets differ; not model selection',
             'nodes_per_endpoint':64000,'counts':dict(counts),'failures':failures}
        (ART/'ranking-failure-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps(out['counts'],indent=2))
        for p in sorted(failures,key=lambda p:p['cp_loss'],reverse=True)[:12]:
            print(json.dumps({k:p[k] for k in ('kind','cp_loss','base_margin_cp','hybrid_margin_cp','cause','good_fen','bad_fen','good_details','bad_details','good_search','bad_search')}))

if __name__=='__main__':main()
