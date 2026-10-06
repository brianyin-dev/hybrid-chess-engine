"""Prefit deeper retention validation and split/history integrity checks."""
import copy,json
from collections import Counter
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from engine.strategic import StrategicEvaluator,PRIORS
from ml.collect_relationship_v35 import ROOT,ART,SPLITS
from ml.run_search_v22 import exact
from ml.generate_search_data import key
from ml.diagnose_critical_v23 import review
from ml.run_balanced_v19 import write,digest,SF

def main():
    if (ART/'training-protocol.json').exists():raise ValueError('Validate before training only')
    if (ART/'data-audit.json').exists():raise FileExistsError('Preserve completed validation')
    path=ART/'collection.json';raw=ART/'collection-before-retention-confirmation.json'
    if not raw.exists():raw.write_bytes(path.read_bytes())
    d=json.loads(raw.read_text());assert d['complete'];removed=[]
    with exclusive_cpu('v35 prefit deeper correct-decision validation'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
        for s in SPLITS:
            retained=[]
            for p in d['pairs'][s]:
                if p['kind']=='retention':
                    root={'initial_fen':chess.STARTING_FEN,'history':p['root_history']};b=reconstruct(root)
                    result=review(sf,b,chess.Move.from_uci(p['heuristic_move']),128000)
                    if result['cp_loss'] is None or result['cp_loss']>=25 or result['allows_mate'] or result['missed_forced_mate']:
                        removed.append({'split':s,'root_fen':p['root_fen'],'game_id':p['game_id'],'review':result});continue
                    p['heuristic_correct_confirmation']=result
                retained.append(p)
            d['pairs'][s]=retained;endpoint_fens={p[f] for p in retained for f in ('good_fen','bad_fen')};d['rows'][s]=[r for r in d['rows'][s] if r['fen'] in endpoint_fens]
            print('Validated',s,len(retained),'pairs',flush=True)
    write(path,d);write(ART/'retention-confirmation.json',{'raw_sha256':digest(raw),'validated_sha256':digest(path),'removed':removed,'timing':'Teacher-only before fitting; reject retention roots where deeper128k analysis finds consequential HCE error or mate inconsistency.'})
    base=StrategicEvaluator(PRIORS);families={};aliases={};stats={};src=json.loads((ART/'sources.json').read_text())
    for s in SPLITS:
        rows=d['rows'][s]+d['ordinary'][s];byfen={r['fen']:r for r in d['rows'][s]}
        for r in rows:
            b=reconstruct(r);assert b.fen()==r['fen'] and b.is_valid() and not b.is_check() and not b.is_game_over();assert base.evaluate_position(b)==r['strategic_base_cp']
        for p in d['pairs'][s]:
            g,z=byfen[p['good_fen']],byfen[p['bad_fen']]
            assert g['game_id']==z['game_id']==p['game_id'] and g['family']==z['family']==p['family']
            assert p['base_margin_cp']==p['sign']*(g['strategic_base_cp']-z['strategic_base_cp'])
            assert -120<p['base_margin_cp']<=100 and p['cp_loss']>=25
            assert p['sign']*(p['endpoint_search'][0]['white_cp']-p['endpoint_search'][1]['white_cp'])>=15
            rb=reconstruct({'initial_fen':chess.STARTING_FEN,'history':p['root_history']});assert rb.fen()==p['root_fen']
            for confirmation in p['root_confirmation']:assert chess.Move.from_uci(confirmation['move']) in rb.legal_moves
            for endpoint,confirmation in zip((g,z),p['endpoint_search']):
                eb=reconstruct(endpoint);move=chess.Move.from_uci(confirmation['best_move']);assert move in eb.legal_moves and not eb.is_capture(move) and not move.promotion and not eb.gives_check(move)
            for r in (g,z):assert r['history'][:len(p['root_history'])]==p['root_history'] and r['history'][len(p['root_history'])]==r['root_move']
            if p['kind']=='repair':assert p['heuristic_move']==z['root_move'] and p['base_margin_cp']<=0
            else:assert p['heuristic_correct_confirmation']['cp_loss']<25 and p['base_margin_cp']>0
        families[s]={g['family'] for g in src['sources'] if g['split']==s};aliases[s]={key(r['fen']) for r in rows+d['screen_roots'][s]}
        stats[s]={'pairs':len(d['pairs'][s]),'kinds':dict(Counter(p['kind'] for p in d['pairs'][s])),'contributing_games':len({p['game_id'] for p in d['pairs'][s]}),'contributing_families':len({p['family'] for p in d['pairs'][s]}),'ordinary_rows':len(d['ordinary'][s]),'source_games':sum(g['split']==s for g in src['sources']),'source_families':len(families[s])}
    cross={a+'-'+b:{'family_overlap':len(families[a]&families[b]),'alias_overlap':len(aliases[a]&aliases[b])} for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(v for row in cross.values() for v in row.values())
    for game in src['sources']:
        b=chess.Board()
        for u in game['history']:b.push_uci(u)
        assert b.fen()==game['final_fen']
    protocol=json.loads((ART/'collection-protocol.json').read_text());assert digest(ROOT/'ml/collect_relationship_v35.py')==protocol['collector_sha256']
    write(ART/'data-audit.json',{'passed':True,'stats':stats,'cross_split':cross,'history_baseline_and_confirmed_pair_invariants':True,'collection_sha256':digest(path),'validator_sha256':digest(ROOT/'ml/validate_relationship_v35.py')})
    print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
