"""User-requested 10-game v32 epoch8 hybrid vs unchanged strategic-v26 heuristic.

This explicitly requested exploratory match does not alter prior acceptance gates.
"""
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,platform,sys
from pathlib import Path
from time import perf_counter
import chess
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import play_game,summarize,opening_board
from engine.search import search
from engine.strategic import StrategicEvaluator,PRIORS
from ml.standpat_hybrid import StandPatHybrid

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/standpat-v32'
OUT=ART/'user-pilot-10-500ms'
OPENINGS=ROOT/'benchmarks/openings-decision-v33-500ms-user-pilot.json'
CHECKPOINT=ART/'epoch-8.pt'

def main():
    with exclusive_cpu('user requested 10-game v32 epoch8 NN vs strategic-v26 at 500ms'):
        OUT.mkdir(exist_ok=False)
        openings=json.loads(OPENINGS.read_text())
        assert len(openings)==5
        known={r['fen'] for split in json.loads((ART/'collection.json').read_text())['rows'].values() for r in split}
        canonical=lambda f:' '.join(f.split()[:4])
        aliases={canonical(f) for f in known}
        assert all(canonical(opening_board(o).fen()) not in aliases for o in openings)
        sources=['engine/search.py','engine/strategic.py','engine/evaluation.py','ml/standpat_hybrid.py',
                 'ml/standpat_features.py','ml/threat_model.py','ml/incremental.py','benchmarks/match.py',
                 'ml/run_standpat_v32_500ms_user_pilot.py']
        report={'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','planned_games':10,
                'config':{'time_ms':500,'depth_cap':64,'max_plies_after_opening':600,'pairs':5,
                          'candidate':'standpat-v32-epoch8','baseline':'strategic-v26','correction_weight':.25,
                          'gate':'noncheck_standpat_v2','use_lmr':True},
                'policy':'Explicit user-requested exploratory match after candidate failed prior gate. No model/search changes. Equal cooperative per-move deadlines, sequential CPU lock, no pondering, automatic draws only, no score adjudication. Ply-limit games unfinished, not draws. Fixed first five starts from the prior v32 250ms match, all independently audited within +0.08 to +0.47 pawns for White. Repeated openings are deliberate; this is not fresh holdout confirmation. Same v32 checkpoint as the earlier250ms match; compare only its first five pairs for identical opening coverage. Same openings/budget as the v33 500ms match for model comparison. Small exploratory sample, no statistical strength claim. Colors swapped; book disabled after prefix. No Elo estimate.',
                'python':sys.version,'platform':platform.platform(),'python_chess':chess.__version__,
                'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'openings_sha256':hashlib.sha256(OPENINGS.read_bytes()).hexdigest(),
                'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                'openings':openings,'opening_overlap_v32_static_labels':0,'games':[]}
        pgns=[];candidate=None;base=None
        def selector(name,board,time_limit,depth_cap):
            start=perf_counter()
            result=search(board,depth=depth_cap,time_limit=time_limit,
                          eval_fn=candidate if name=='current' else base,use_lmr=True)
            stats=asdict(result);stats.pop('move');stats['elapsed']=perf_counter()-start
            stats['budget_seconds']=time_limit;stats['overrun_seconds']=max(0,stats['elapsed']-time_limit)
            return result.move,stats
        def save():
            report['summary']=summarize(report['games'])
            report['mean_depth']={name:sum(m['depth'] for g in report['games'] for m in g['moves'] if m['engine']==name)/max(1,sum(m['engine']==name for g in report['games'] for m in g['moves'])) for name in ['current','new_heuristic']}
            for name,content in [('report.json',json.dumps(report,indent=2)+'\n'),('games.pgn',''.join(pgns))]:
                temp=OUT/(name+'.tmp');temp.write_text(content);temp.replace(OUT/name)
        save()
        try:
            for i,o in enumerate(openings,1):
                colors=(chess.WHITE,chess.BLACK) if i%2 else (chess.BLACK,chess.WHITE)
                for color in colors:
                    candidate=StandPatHybrid(CHECKPOINT);base=StrategicEvaluator(PRIORS)
                    print(f"Game {len(report['games'])+1}/10: {o['name']}; NN {'White' if color else 'Black'}",flush=True)
                    r,pgn=play_game(o,color,.5,600,64,selector,i,'new_heuristic')
                    pgn=pgn.replace('[White "current"]','[White "standpat-v32-epoch8"]')
                    pgn=pgn.replace('[Black "current"]','[Black "standpat-v32-epoch8"]')
                    pgns.append(pgn);report['games'].append(r);save()
                    print(json.dumps({'game':len(report['games']),'result':r['current_result'],'termination':r['reason'],'plies':r['played_plies'],'summary':report['summary']}),flush=True)
                    if r['reason'] in ('engine_error','interrupted'):raise RuntimeError('Stopped due to match error/interruption; results preserved')
        except BaseException:
            report['status']='interrupted_or_error';save();raise
        else:report['status']='completed'
        report['finished_at']=datetime.now(timezone.utc).isoformat();save()
        print(json.dumps({'summary':report['summary'],'mean_depth':report['mean_depth']},indent=2),flush=True)

if __name__=='__main__':main()
