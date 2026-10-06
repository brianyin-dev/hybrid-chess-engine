"""Post-match loss diagnosis; frozen models, full histories, no training."""
import json,hashlib
from pathlib import Path
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.analyze import score_data
from ml.diagnose_critical_v23 import review
from ml.run_search_v22 import exact
from ml.run_balanced_v19 import SF
from ml.relationship_hybrid import RelationshipHybrid
from engine.strategic import StrategicEvaluator,PRIORS
from engine.search import search
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ml/artifacts/relationship-v35/user-pilot-20-250ms'
CP=ROOT/'ml/artifacts/relationship-v35/current-epoch-40.pt'
VALUES={chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}
def material(b,color):
    return sum(v*(len(b.pieces(p,color))-len(b.pieces(p,not color))) for p,v in VALUES.items())
def line(sf,b,move=None):
    info=exact(sf,b,[move] if move else None,128000);z=b.copy(stack=True);steps=[]
    for mv in info['pv'][:12]:
        piece=z.piece_at(mv.to_square)
        steps.append({'uci':mv.uci(),'san':z.san(mv),'captured':piece.symbol() if piece else None,'promotion':mv.promotion})
        z.push(mv);steps[-1]['material_cp_for_root_mover']=material(z,b.turn)
    return {'score':score_data(info['score'],b.turn),'depth':info.get('depth'),'pv':steps}
def main():
    target=OUT/'loss-diagnosis.json'
    if target.exists():raise FileExistsError(target)
    report=json.loads((OUT/'report.json').read_text())
    data={'source_sha256':hashlib.sha256((OUT/'report.json').read_bytes()).hexdigest(),
          'policy':'All NN moves in all 11 losses screened at 16000 Stockfish nodes; moves with >=100cp regret, mate or -150 crossing confirmed at128000. Full histories, full strength one thread. First competitive major error (>=150cp or crossing with >=50cp or mate) and worst competitive error receive fixed-depth3 and250ms frozen evaluator probes and128k PVs. Screen scores are approximate and not all moves deeply confirmed. Diagnosis only; no model changes.',
          'games':[]}
    def save():target.write_text(json.dumps(data,indent=2)+'\n')
    with exclusive_cpu('v35 loss diagnosis'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
        for gi,g in enumerate(report['games'],1):
            if g['current_result']!='loss':continue
            b=chess.Board(g['initial_fen'])
            for u in g['opening_moves']:b.push_uci(u)
            rows=[];boards={}
            for ply,m in enumerate(g['moves'],1):
                mv=chess.Move.from_uci(m['uci'])
                if m['engine']=='current':
                    q=review(sf,b,mv,16000);confirmed=False
                    if (q['cp_loss'] or 0)>=100 or q['allows_mate'] or q['missed_forced_mate'] or q['crosses_minus_150']:
                        q=review(sf,b,mv,128000);confirmed=True
                    row={'ply':ply,'fullmove':b.fullmove_number,'san':m['san'],'uci':m['uci'],'fen':b.fen(),'played_depth':m['depth'],'recorded_score':m['score'],'review':q,'confirmed':confirmed,'material_cp':material(b,b.turn)}
                    rows.append(row)
                    if confirmed and q['qualifies']:boards[ply]=b.copy(stack=True)
                b.push(mv)
            major=[r for r in rows if r['confirmed'] and r['review']['qualifies']]
            chosen=list({r['ply']:r for r in ([major[0],max(major,key=lambda r:r['review']['cp_loss'] or 100000)] if major else [])}.values())
            cases=[]
            for r in chosen:
                b=boards[r['ply']];case={'ply':r['ply'],'best_line':line(sf,b),'played_line':line(sf,b,chess.Move.from_uci(r['uci'])),'probes':{}}
                for mode,opts in [('depth3',{'depth':3}),('250ms',{'depth':64,'time_limit':.25})]:
                    probes={}
                    for name,model in [('heuristic',StrategicEvaluator(PRIORS)),('nn',RelationshipHybrid(CP))]:
                        p=search(b.copy(stack=True),eval_fn=model,use_lmr=True,**opts)
                        probes[name]={'move':p.move.uci(),'san':b.san(p.move),'depth':p.depth,'nodes':p.nodes,'elapsed':p.elapsed,'score':p.score}
                    for name,p in probes.items():p['review']=review(sf,b,chess.Move.from_uci(p['move']),128000)
                    case['probes'][mode]=probes
                cases.append(case)
            item={'game':gi,'color':g['current_color'],'opening':g['opening'],'plies':g['played_plies'],'nn_moves':rows,'confirmed_major_competitive_errors':len(major),'first_major_ply':major[0]['ply'] if major else None,'critical_cases':cases}
            data['games'].append(item);save()
            print(json.dumps({'game':gi,'major_errors':len(major),'first':{k:major[0][k] for k in ['ply','san','played_depth','review']} if major else None}),flush=True)
    data['status']='completed';save()
if __name__=='__main__':main()
