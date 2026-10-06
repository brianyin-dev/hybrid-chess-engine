"""Find and trace the first confirmed competitive error in each v32 500ms loss."""
import json
from pathlib import Path
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from ml.diagnose_critical_v23 import review,forced
from ml.run_balanced_v19 import write,digest,SF
from ml.standpat_hybrid import StandPatHybrid
from engine.strategic import StrategicEvaluator,PRIORS
from engine.search import search
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'ml/artifacts/evaluator-v34'
SOURCE=ROOT/'ml/artifacts/standpat-v32/user-pilot-10-500ms/report.json'
CP=ROOT/'ml/artifacts/standpat-v32/epoch-8.pt'
def main():
    ART.mkdir(exist_ok=True)
    target=ART/'loss-diagnosis.json'
    if target.exists():raise FileExistsError('Preserve recorded diagnosis')
    data={'source_sha256':digest(SOURCE),'checkpoint_sha256':digest(CP),'policy':'First competitive error per lost game; 16k screening, 128k confirmation; full game histories. Fixed depth3 probes and forced full-window depth3 score-bearing leaves. Diagnosis only, repeated games not holdout.', 'games':[]}
    with exclusive_cpu('v34 evaluator loss diagnosis'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32})
        models={'heuristic':StrategicEvaluator(PRIORS),'nn':StandPatHybrid(CP)}
        for gi,g in enumerate(json.loads(SOURCE.read_text())['games'],1):
            if g['current_result']!='loss':continue
            b=chess.Board(g['initial_fen'])
            for u in g['opening_moves']:b.push_uci(u)
            failure=None
            for ply,m in enumerate(g['moves'],1):
                mv=chess.Move.from_uci(m['uci'])
                if m['engine']=='current':
                    check=review(sf,b,mv,16000)
                    if check['qualifies']:
                        check=review(sf,b,mv,128000)
                        if check['qualifies']:
                            failure={'fen':b.fen(),'initial_fen':g['initial_fen'],'history':[u.uci() for u in b.move_stack],'played':mv.uci(),'ply':ply,'review':check,'probes':{},'traces':{}}
                            for name,model in models.items():
                                p=search(b.copy(stack=True),depth=3,eval_fn=model,use_lmr=True)
                                failure['probes'][name]={'move':p.move.uci(),'depth':p.depth,'score':p.score,'nodes':p.nodes,'review':review(sf,b,p.move,128000)}
                            moves={mv.uci(),check['best_move'],*[p['move'] for p in failure['probes'].values()]}
                            for u in sorted(moves):
                                traces={name:forced(b.copy(stack=True),chess.Move.from_uci(u),model) for name,model in models.items()}
                                for t in traces.values():
                                    leaf=t.get('leaf')
                                    if leaf:
                                        lb=chess.Board(leaf['fen']);leaf['baseline_cp']=models['heuristic'].evaluate_position(lb);leaf['nn_cp']=models['nn'].evaluate_position(lb);leaf['correction_cp']=leaf['nn_cp']-leaf['baseline_cp']
                                failure['traces'][u]=traces
                            break
                b.push(mv)
            data['games'].append({'game':gi,'pair':g['pair'],'failure':failure});write(target,data)
            print('Diagnosed',gi,'played',failure['played'] if failure else None,'choices',{n:p['move'] for n,p in failure['probes'].items()} if failure else {},flush=True)
if __name__=='__main__':main()
