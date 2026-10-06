"""Supplemental fixed-model Black-to-move coverage audit, never model selection."""
import json,copy,random
from collections import Counter
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from ml.collect_relationship_v35 import ROOT,ART
from ml.generate_search_data import key
from ml.run_balanced_v19 import write,digest,SF
from ml.run_search_v22 import exact
from ml.train_relationship_v35 import screen
from ml.relationship_hybrid import RelationshipHybrid
from ml.standpat_hybrid import StandPatHybrid
from engine.strategic import StrategicEvaluator,PRIORS

def main():
    if (ART/'black-depth3.json').exists():raise FileExistsError('Preserve supplemental coverage')
    selection=json.loads((ART/'selection.json').read_text());decision=json.loads((ART/'decision.json').read_text())
    write(ART/'black-coverage-protocol.json',{'reason':'Final audit found original development/test searched roots all White to move. Training includes42 Black root-ranking pairs and170 Black endpoints. Add a fixed-model Black cohort from reserved test games without changing selection, weights, gates or adoption.','timing':'After original selection/test; diagnostic supplement, not a second independent game sample. Games/families overlap original test cohort, never training. No model/score-based case selection. Teacher32k abs<=800 prefilter mirrors original root filter.','selected':selection['selected'],'selected_sha256':{a:digest(ART/p) for a,p in selection['selected'].items()},'original_decision':decision,'selection_can_change':False})
    d=json.loads((ART/'collection.json').read_text());src=json.loads((ART/'sources.json').read_text())
    forbidden={key(r['fen']) for s in ('train','val','test') for r in d['rows'][s]+d['ordinary'][s]+d['screen_roots'][s]}
    candidates=[]
    for game in src['sources']:
        if game['split']!='test':continue
        prefix=len(game['opening']['moves'])
        for offset in (21,45,69,93,117,141):
            end=prefix+offset
            if end>=len(game['history']):continue
            r={'initial_fen':chess.STARTING_FEN,'history':game['history'][:end],'game_id':game['game_id'],'family':game['family'],'split':'test','source_ply':offset}
            b=reconstruct(r);r['fen']=b.fen();r['screen_stage']='endgame' if len(b.piece_map())<=12 else 'middlegame'
            if b.turn==chess.BLACK and not b.is_game_over() and key(b.fen()) not in forbidden:candidates.append(r)
    random.Random(350040).shuffle(candidates);selected=[];games=set();families=Counter();stages=Counter()
    with exclusive_cpu('v35 supplemental fixed Black coverage'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
        while len(selected)<24:
            options=[r for r in candidates if r['game_id'] not in games and families[r['family']]<2]
            if not options:raise ValueError('Insufficient Black cohort')
            r=min(options,key=lambda r:(families[r['family']],stages[r['screen_stage']]))
            candidates.remove(r);teacher=exact(sf,reconstruct(r),nodes=32000);cp=teacher['score'].white().score()
            if cp is None or abs(cp)>800:continue
            selected.append({**r,'teacher_root_white_cp':cp});games.add(r['game_id']);families[r['family']]+=1;stages[r['screen_stage']]+=1
        write(ART/'black-roots.json',selected)
        models={'heuristic':StrategicEvaluator(PRIORS),'old_v32':StandPatHybrid(ROOT/'ml/artifacts/standpat-v32/epoch-8.pt'),**{a:RelationshipHybrid(ART/p) for a,p in selection['selected'].items()}}
        screen(sf,selected,models,'black-depth3','depth3');screen(sf,selected,models,'black-250ms','250ms')
    assert json.loads((ART/'selection.json').read_text())==selection and json.loads((ART/'decision.json').read_text())==decision
    write(ART/'black-coverage-audit.json',{'passed':True,'positions':24,'unique_games':len(games),'families':len(families),'all_black_to_move':True,'selection_and_decision_unchanged':True,'overlap_with_training_families':len(set(families)&{g['family'] for g in src['sources'] if g['split']=='train'}),'cohort_shares_reserved_test_games_not_independent_of_white_cohort':True})
if __name__=='__main__':main()
