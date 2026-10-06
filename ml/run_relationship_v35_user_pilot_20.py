"""User-requested 20-game v35 current-input epoch40 hybrid vs unchanged strategic-v26 heuristic.

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
from ml.relationship_hybrid import RelationshipHybrid
from ml.generate_search_data import key

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/relationship-v35'
OUT=ART/'user-pilot-20-250ms'
OPENINGS=ROOT/'benchmarks/openings-relationship-v35-user-pilot-extension.json'
CHECKPOINT=ART/'current-epoch-40.pt'

def main():
    with exclusive_cpu('user requested 20-game v35 current-input NN vs strategic-v26 at 250ms'):
        prior_path=ART/'user-pilot-10-250ms/report.json'
        prior=json.loads(prior_path.read_text())
        assert prior['status']=='completed' and len(prior['games'])==10
        assert prior['checkpoint_sha256']==hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        for p,h in prior['source_sha256'].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
        OUT.mkdir(exist_ok=False)
        openings=json.loads(OPENINGS.read_text())
        assert len(openings)==5
        collection=json.loads((ART/'collection.json').read_text())
        aliases={key(r['fen']) for field in ('rows','ordinary','screen_roots') for rs in collection[field].values() for r in rs}
        assert all(key(opening_board(o).fen()) not in aliases for o in openings)
        sources=['engine/search.py','engine/strategic.py','engine/evaluation.py','ml/relationship_hybrid.py',
                 'ml/relationship_features.py','ml/standpat_features.py','ml/threat_model.py','ml/incremental.py',
                 'benchmarks/match.py','ml/run_relationship_v35_user_pilot.py','ml/run_relationship_v35_user_pilot_20.py']
        report={'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','planned_games':20,
                'config':{'time_ms':250,'depth_cap':64,'max_plies_after_opening':600,'pairs':10,
                          'candidate':'relationship-v35-current-epoch40','baseline':'strategic-v26','correction_weight':.25,
                          'gate':'noncheck_standpat_v2','use_lmr':True},
                'policy':'Explicit user-requested exploratory match after candidate failed original position gate. Frozen current-input epoch40, not augmented-feature model. Both sides250ms; unchanged common search/LMR; sequential CPU lock, no pondering, automatic draws only, no score adjudication or hard clock forfeits. Ply-limit games unfinished, not draws. Ten weighted standard-book starts (five retained, five fresh), each played with both colors; book disabled after prefix. No claim all families are historically unseen. Model/search unchanged, no tuning or automatic expansion. No Elo estimate.',
                'python':sys.version,'platform':platform.platform(),'python_chess':chess.__version__,
                'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'openings_sha256':hashlib.sha256(OPENINGS.read_bytes()).hexdigest(),
                'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                'openings':prior['openings']+openings,'opening_overlap_v35_labels_and_roots':0,
                'prior_report_sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                'prior_report_path':str(prior_path.relative_to(ROOT)),
                'extension_policy':'User requested extension after seeing first ten results; adaptive exploratory total, not a predeclared independent 20-game experiment.',
                'games':prior['games'].copy()}
        pgns=[(ART/'user-pilot-10-250ms/games.pgn').read_text().rstrip()+'\n\n'];candidate=None;base=None
        def selector(name,board,time_limit,depth_cap):
            start=perf_counter()
            result=search(board,depth=depth_cap,time_limit=time_limit,
                          eval_fn=candidate if name=='current' else base,use_lmr=True)
            stats=asdict(result);stats.pop('move');stats['elapsed']=perf_counter()-start
            stats['budget_seconds']=time_limit;stats['overrun_seconds']=max(0,stats['elapsed']-time_limit)
            return result.move,stats
        def save():
            report['summary']=summarize(report['games'])
            report['extension_summary']=summarize(report['games'][10:])
            report['mean_depth']={name:sum(m['depth'] for g in report['games'] for m in g['moves'] if m['engine']==name)/max(1,sum(m['engine']==name for g in report['games'] for m in g['moves'])) for name in ['current','new_heuristic']}
            for name,content in [('report.json',json.dumps(report,indent=2)+'\n'),('games.pgn',''.join(pgns))]:
                temp=OUT/(name+'.tmp');temp.write_text(content);temp.replace(OUT/name)
        save()
        try:
            for i,o in enumerate(openings,6):
                colors=(chess.WHITE,chess.BLACK) if i%2 else (chess.BLACK,chess.WHITE)
                for color in colors:
                    candidate=RelationshipHybrid(CHECKPOINT);base=StrategicEvaluator(PRIORS)
                    print(f"Game {len(report['games'])+1}/20: {o['name']}; NN {'White' if color else 'Black'}",flush=True)
                    r,pgn=play_game(o,color,.25,600,64,selector,i,'new_heuristic')
                    pgn=pgn.replace('[White "current"]','[White "relationship-v35-current-epoch40"]')
                    pgn=pgn.replace('[Black "current"]','[Black "relationship-v35-current-epoch40"]')
                    pgns.append(pgn);report['games'].append(r);save()
                    print(json.dumps({'game':len(report['games']),'result':r['current_result'],'termination':r['reason'],'plies':r['played_plies'],'summary':report['summary']}),flush=True)
                    if r['reason'] in ('engine_error','interrupted'):raise RuntimeError('Stopped due to match error/interruption; results preserved')
        except BaseException:
            report['status']='interrupted_or_error';save();raise
        else:report['status']='completed'
        report['finished_at']=datetime.now(timezone.utc).isoformat();save()
        print(json.dumps({'summary':report['summary'],'mean_depth':report['mean_depth']},indent=2),flush=True)

if __name__=='__main__':main()
