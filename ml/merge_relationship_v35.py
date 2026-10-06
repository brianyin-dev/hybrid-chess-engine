"""Reuse existing settled labels only when they match actual heuristic decisions."""
import copy,json
from collections import Counter
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from engine.search import search
from engine.strategic import StrategicEvaluator,PRIORS
from ml.collect_relationship_v35 import ROOT,ART,OLD,SPLITS
from ml.generate_search_data import key
from ml.diagnose_critical_v23 import review
from ml.run_balanced_v19 import write,digest,SF

def main():
    if (ART/'training-protocol.json').exists():raise ValueError('Merge before fitting only')
    if (ART/'prior-pair-merge.json').exists():raise FileExistsError('Preserve merge')
    path=ART/'collection.json';d=json.loads(path.read_text());assert d['complete']
    raw=ART/'collection-before-prior-pair-merge.json';raw.write_bytes(path.read_bytes())
    previous=json.loads((OLD/'collection.json').read_text());base=StrategicEvaluator(PRIORS);rejected=Counter();added=Counter()
    used={key(r['fen']) for s in SPLITS for r in d['rows'][s]+d['ordinary'][s]+d['screen_roots'][s]}
    with exclusive_cpu('v35 teacher-only reuse of verified independent decision labels'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
        for s in SPLITS:
            byfen={r['fen']:r for r in previous['rows'][s]}
            for p in previous['pairs'][s]:
                if p['base_margin_cp']<=-120:rejected['new_feasibility_safety_margin']+=1;continue
                if any(key(p[f]) in used for f in ('good_fen','bad_fen')):rejected['duplicate']+=1;continue
                b=reconstruct({'initial_fen':chess.STARTING_FEN,'history':p['root_history']})
                h=search(b.copy(stack=True),depth=2,node_limit=12000,eval_fn=base,use_lmr=True)
                bad=byfen[p['bad_fen']]['root_move']
                if p['kind']=='repair' and h.move.uci()!=bad:rejected['repair_not_actual_heuristic_choice']+=1;continue
                confirmation=review(sf,b,h.move,128000)
                if p['kind']=='repair':
                    if confirmation['cp_loss'] is None or confirmation['cp_loss']<25:rejected['no_actual_root_mistake']+=1;continue
                elif confirmation['cp_loss'] is None or confirmation['cp_loss']>=25 or confirmation['allows_mate'] or confirmation['missed_forced_mate']:
                    rejected['retention_not_correct_heuristic_decision']+=1;continue
                q=copy.deepcopy(p);q.update(trace_depth=2,heuristic_move=h.move.uci(),heuristic_depth_completed=h.depth,heuristic_root_regret_screen_cp=confirmation['cp_loss'],reused_prior_pair=True,label_kind='confirmed_search_decision_qleaf_rank_v35')
                if q['kind']=='retention':q['heuristic_correct_confirmation']=confirmation
                endpoints=[]
                for f in ('good_fen','bad_fen'):
                    row=copy.deepcopy(byfen[q[f]]);row['trace_depth']=2;row['reused_prior_pair']=True;endpoints.append(row)
                d['pairs'][s].append(q);d['rows'][s]+=endpoints;used.update(key(r['fen']) for r in endpoints);added[s]+=1
            print('Reused verified pairs',s,added[s],flush=True)
    write(path,d);write(ART/'prior-pair-merge.json',{'timing':'Before fitting; reuse Stockfish labels, not old model predictions. Existing v33 whole-family splits retained. Each prior repair must be the actual depth2 heuristic choice and have >=25cp confirmed root loss; prior retention must remain correct within25cp under fresh128k root analysis.','source_sha256':digest(OLD/'collection.json'),'collector_raw_sha256':digest(raw),'merged_sha256':digest(path),'added_pairs':dict(added),'rejected':dict(rejected),'utility_sha256':digest(ROOT/'ml/merge_relationship_v35.py')})
if __name__=='__main__':main()
