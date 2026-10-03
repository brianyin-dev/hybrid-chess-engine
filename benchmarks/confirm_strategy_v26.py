"""100 paired equal-time games only after the untouched position test passes."""
import hashlib,json
from pathlib import Path
import chess
from dataclasses import asdict
from benchmarks.build_strategy_v26 import ROOT,ART,save
from benchmarks.match import play_game,summarize,opening_board
from benchmarks.cpu_lock import exclusive_cpu
from engine.evaluation import evaluate
from engine.search import search
from engine.strategic import StrategicEvaluator
from ml.generate_search_data import key

def main():
    decision=json.loads((ART/'decision.json').read_text())
    if not decision['qualifies_for_100_games']:raise ValueError('Unseen searched-choice gate failed; do not run games')
    selection=json.loads((ART/'selection.json').read_text());weights=selection['weights']
    path=ROOT/('benchmarks/openings-strategy-v26-corrected-confirmation.json'
               if (ART/'tempo-correction-protocol.json').exists() else 'benchmarks/openings-strategy-v26-confirmation.json')
    openings=json.loads(path.read_text())
    forbidden={key(r['fen']) for r in json.loads((ART/'training-labels.json').read_text())}
    for s in ('train','val','test'):forbidden.update(key(r['fen']) for r in json.loads((ART/f'{s}-roots.json').read_text()))
    assert not any(key(opening_board(o).fen()) in forbidden for o in openings),'Confirmation start overlaps dataset'
    out=ART/'confirmation';out.mkdir(exist_ok=True)
    report={'status':'running','games':[],'budget_seconds':.25,'depth_cap':64,'planned_games':100,'openings_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'weights':weights,'selected':selection['selected'],'baseline':'Original v24 heuristic plusunchanged LMR; candidate only changes evaluator. No NN orbookafterprefix.'}
    if (out/'report.json').exists():report=json.loads((out/'report.json').read_text())
    pgns=(out/'games.pgn').read_text() if (out/'games.pgn').exists() else ''
    def selector(name,b,time_limit,depth_cap):
        model=StrategicEvaluator(weights) if name=='current' else evaluate
        p=search(b,depth=depth_cap,time_limit=time_limit,eval_fn=model,use_lmr=True)
        stats=asdict(p);stats.pop('move');stats.update(budget_seconds=time_limit,overrun_seconds=max(0,p.elapsed-time_limit))
        return p.move,stats
    with exclusive_cpu('v26 100fresh paired confirmationgames'):
        for i,o in enumerate(openings,1):
            for color in (chess.WHITE,chess.BLACK):
                if any(g['pair']==i and g['current_color']==('white' if color else 'black') for g in report['games']):continue
                g,pgn=play_game(o,color,.25,600,64,selector,i,'baseline');report['games'].append(g);pgns+=pgn
                report['summary']=summarize(report['games']);report['status']='completed' if len(report['games'])==100 else 'running'
                (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'games.pgn').write_text(pgns)
                s=report['summary'];print('Confirmation',len(report['games']),'/100',s['wins'],'W',s['draws'],'D',s['losses'],'L',flush=True)
                if s['errors']:raise RuntimeError('Game error; preserve output and investigate')
    s=report['summary'];passed=s['completed']==100 and not s['errors'] and s['score_fraction_completed']>.5
    decision.update(confirmation_passed=passed,confirmation_summary=s,app_promoted=False)
    save('decision.json',decision)

if __name__=='__main__':main()
