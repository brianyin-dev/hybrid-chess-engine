"""Verify provenance and exact supervision invariants without model selection."""
import json
from collections import Counter
import chess
from benchmarks.build_strategy_v26 import reconstruct
from engine.strategic import PRIORS,StrategicEvaluator
from ml.generate_search_data import key
from ml.run_balanced_v19 import write,digest
from ml.run_decision_v33 import ART,ROOT,OLD,INITIAL

def main():
    d=json.loads((ART/'collection.json').read_text());sources=json.loads((ART/'sources.json').read_text());protocol=json.loads((ART/'protocol.json').read_text())
    base=StrategicEvaluator(PRIORS);familysets={};aliassets={};stats={}
    for s in ('train','val','test'):
        rows=d['rows'][s];byfen={r['fen']:r for r in rows};roots=[r for r in sources['roots'][s]]
        for r in roots+rows:
            b=reconstruct(r);assert b.fen()==r['fen'] and b.is_valid()
            if r in rows:
                assert not b.is_check() and not b.is_game_over()
                assert base(b)==r['strategic_base_cp']
        familysets[s]={g['family'] for g in sources['sources'] if g['split']==s}
        aliassets[s]={key(r['fen']) for r in rows+roots}
        for p in d['pairs'][s]:
            g,z=byfen[p['good_fen']],byfen[p['bad_fen']]
            assert p['game_id']==g['game_id']==z['game_id']
            assert p['family']==g['family']==z['family']
            assert p['sign']*(g['score_cp']-z['score_cp'])==p['static_gap_cp']>=15
            assert p['sign']*(g['strategic_base_cp']-z['strategic_base_cp'])==p['base_margin_cp']
            assert -125<p['base_margin_cp']<=100
            assert p['cp_loss']==p['root_confirmation'][0]['mover_cp']-p['root_confirmation'][1]['mover_cp']>=25
            assert p['sign']*(p['endpoint_search'][0]['white_cp']-p['endpoint_search'][1]['white_cp'])>=15
            for endpoint in (g,z):
                h=len(p['root_history']);assert endpoint['history'][:h]==p['root_history']
                assert endpoint['history'][h]==endpoint['root_move']
                a=p['endpoint_search'][0 if endpoint is g else 1];b=reconstruct(endpoint);m=chess.Move.from_uci(a['best_move'])
                assert m in b.legal_moves and not b.is_capture(m) and not m.promotion and not b.gives_check(m)
        stats[s]={'pairs':len(d['pairs'][s]),'endpoint_count':len(rows),'source_games':sum(g['split']==s for g in sources['sources']),
                  'contributing_games':len({p['game_id'] for p in d['pairs'][s]}),'contributing_families':len({p['family'] for p in d['pairs'][s]}),
                  'tags':dict(Counter(t for p in d['pairs'][s] for t in p['tags'])),
                  'tag_games':{t:len({p['game_id'] for p in d['pairs'][s] if t in p['tags']}) for t in ['advanced_pawn','pin','king_pressure','immediate_promotion']}}
    cross={a+'-'+b:{'families':len(familysets[a]&familysets[b]),'position_aliases':len(aliassets[a]&aliassets[b])} for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(n for r in cross.values() for n in r.values()),cross
    for g in sources['sources']:
        b=chess.Board()
        for move in g['history']:b.push_uci(move)
        assert b.fen()==g['final_fen']
    hashes={p:digest(ROOT/p)==sha for p,sha in protocol['source_sha256'].items()}
    assert all(hashes.values()) and digest(INITIAL)==protocol['initial_sha256']
    write(ART/'data-audit.json',{'passed':True,'stats':stats,'cross_split':cross,'protocol_hashes_unchanged':hashes,
                               'full_histories_and_static_baselines_verified':True,'root_and_leaf_rankings_verified':True,
                               'search_settling_filters_verified':True,'known_prior_alias_policy':'Collector excludes prior canonical label/match aliases; legacy warm provenance inherited.'})
    print(json.dumps({'stats':stats,'cross_split':cross},indent=2))
if __name__=='__main__':main()
