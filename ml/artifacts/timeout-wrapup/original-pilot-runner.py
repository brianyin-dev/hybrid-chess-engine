"""User-requested 20-game corrected timeout vs frozen previous search, identical v35 hybrid.

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
from benchmarks.pre_timeout.search import search as previous_search
from engine.strategic import StrategicEvaluator,PRIORS
from ml.relationship_hybrid import RelationshipHybrid
from ml.generate_search_data import key

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/relationship-v35'
OUT=ROOT/'ml/artifacts/timeout-wrapup/pilot-20'
OPENINGS=ROOT/'benchmarks/openings-timeout-wrapup.json'
CHECKPOINT=ART/'current-epoch-40.pt'

def main():
    with exclusive_cpu('20-game timeout fix vs frozen search, identical v35 evaluator at250ms'):
        OUT.mkdir(exist_ok=False)
        openings=json.loads(OPENINGS.read_text())
        assert len(openings)==10
        collection=json.loads((ART/'collection.json').read_text())
        aliases={key(r['fen']) for field in ('rows','ordinary','screen_roots') for rs in collection[field].values() for r in rs}
        assert all(key(opening_board(o).fen()) not in aliases for o in openings)
        sources=['engine/search.py','engine/strategic.py','engine/evaluation.py','ml/relationship_hybrid.py',
                 'ml/relationship_features.py','ml/standpat_features.py','ml/threat_model.py','ml/incremental.py',
                 'benchmarks/match.py','ml/run_timeout_wrapup_pilot.py','benchmarks/pre_timeout/search.py']
        report={'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','planned_games':20,
                'config':{'time_ms':250,'depth_cap':64,'max_plies_after_opening':600,'pairs':10,
                          'candidate':'timeout-fixed-v35','baseline':'pre-timeout-v35','correction_weight':.25,
                          'gate':'noncheck_standpat_v2','use_lmr':True},
                'policy':'Twenty predeclared pilot games: same frozen current-input v35 epoch40 checkpoint for both sides,25% correction. Only timeout search differs; comparator is exact archived pre-fix source.250ms, LMR,depthcap64,CPU lock,no pondering,automatic draws only,no score adjudication or hard clock forfeits.600ply limit unfinished, not draws.10 fresh book starts played with both colors; book disabled after prefix. Sequential paired games; clock variability remains. Screen starts separate; not all opening families historically unseen. No rating or NN-adoption claim.',
                'python':sys.version,'platform':platform.platform(),'python_chess':chess.__version__,
                'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'openings_sha256':hashlib.sha256(OPENINGS.read_bytes()).hexdigest(),
                'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                'openings':openings,'opening_overlap_v35_labels_and_roots':0,'games':[]}
        pgns=[];candidate=None;base=None
        def selector(name,board,time_limit,depth_cap):
            start=perf_counter()
            result=(search if name=='current' else previous_search)(board,depth=depth_cap,time_limit=time_limit,
                          eval_fn=candidate if name=='current' else base,use_lmr=True)
            stats=asdict(result);stats.pop('move');stats['elapsed']=perf_counter()-start
            stats['budget_seconds']=time_limit;stats['overrun_seconds']=max(0,stats['elapsed']-time_limit)
            return result.move,stats
        def save():
            report['summary']=summarize(report['games'])
            report['mean_depth']={name:sum(m['depth'] for g in report['games'] for m in g['moves'] if m['engine']==name)/max(1,sum(m['engine']==name for g in report['games'] for m in g['moves'])) for name in ['current','pre_timeout']}
            for name,content in [('report.json',json.dumps(report,indent=2)+'\n'),('games.pgn',''.join(pgns))]:
                temp=OUT/(name+'.tmp');temp.write_text(content);temp.replace(OUT/name)
        save()
        try:
            for i,o in enumerate(openings,1):
                colors=(chess.WHITE,chess.BLACK) if i%2 else (chess.BLACK,chess.WHITE)
                for color in colors:
                    candidate=RelationshipHybrid(CHECKPOINT);base=RelationshipHybrid(CHECKPOINT)
                    print(f"Game {len(report['games'])+1}/20: {o['name']}; NN {'White' if color else 'Black'}",flush=True)
                    r,pgn=play_game(o,color,.25,600,64,selector,i,'pre_timeout')
                    pgn=pgn.replace('[White "current"]','[White "timeout-fixed-v35"]')
                    pgn=pgn.replace('[Black "current"]','[Black "timeout-fixed-v35"]')
                    pgn=pgn.replace('pre_timeout','pre-timeout-v35')
                    pgns.append(pgn);report['games'].append(r);save()
                    print(json.dumps({'game':len(report['games']),'result':r['current_result'],'termination':r['reason'],'plies':r['played_plies'],'summary':report['summary']}),flush=True)
                    if r['reason'] in ('engine_error','interrupted'):raise RuntimeError('Stopped due to match error/interruption; results preserved')
        except BaseException:
            report['status']='interrupted_or_error';save();raise
        else:report['status']='completed'
        report['finished_at']=datetime.now(timezone.utc).isoformat();save()
        print(json.dumps({'summary':report['summary'],'mean_depth':report['mean_depth']},indent=2),flush=True)

if __name__=='__main__':main()
