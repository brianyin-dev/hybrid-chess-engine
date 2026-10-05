"""Static-label provenance, instrumentation parity, gate coverage and inference."""
import io
import json
from collections import Counter
import chess
import chess.engine
import chess.pgn
import torch
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from engine.search import _Search, _SearchLimit, INF, _position_key
from engine.strategic import PRIORS, StrategicEvaluator
from ml.run_balanced_v19 import write, digest, SF
from ml.run_standpat_v32 import ART, ROOT, PREVIOUS, StaticSampler, known_provenance, inference_audit, gate
from ml.run_search_correction_v29 import family
from ml.generate_search_data import key
from ml.static_teacher import parse_static_line, static_evaluation
from ml.standpat_hybrid import StandPatHybrid, gate_active
from ml.train import attainable_margin_cp


def main():
    with exclusive_cpu('v32 final static-label provenance and inference audit'):
        torch.set_num_threads(1)
        data=json.loads((ART/'collection.json').read_text()); assert data['complete']
        roots=json.loads((ART/'fresh-roots.json').read_text())
        pairs=json.loads((ART/'pairs.json').read_text())
        retention=json.loads((ART/'ordinary-retention.json').read_text())
        known,known_families=known_provenance()
        sources={r['game_id']:r for r in data['sources']}
        for s in sources.values():
            b=chess.Board()
            for u in s['history']:
                b.push_uci(u)
            assert b.fen()==s['final_fen'] and family(s['history'])==s['family']
            assert s['family'] not in known_families
        aliases={}; families={}; counts={}; root_counts={}; pair_endpoints=set()
        base=StrategicEvaluator(PRIORS)
        for split,rows in data['rows'].items():
            aliases[split]={key(r['fen']) for r in rows}
            assert len(aliases[split])==len(rows) and not aliases[split]&known
            families[split]={r['family'] for r in data['roots'][split]}
            for r in rows:
                root=data['roots'][split][r['root_index']]; source=sources[r['game_id']]
                assert r['split']==split==source['split'] and r['family']==root['family']==source['family']
                assert r['history'][:len(root['history'])]==root['history']
                b=reconstruct(r)
                assert b.fen()==r['fen'] and gate_active(b) and not b.is_game_over(claim_draw=False)
                assert r['label_kind']=='stockfish_static_white_cp' and parse_static_line(r['teacher_final_line'])==r['score_cp']
                assert r['strategic_base_cp']==base(b)
                category='capture' if next(b.generate_legal_captures(),None) is not None else 'quiet'
                assert r['category']==category and r['search_ply']==len(r['history'])-len(root['history'])
            byfen={r['fen']:r for r in rows}
            for p in pairs[split]:
                g,z=byfen[p['good_fen']],byfen[p['bad_fen']]
                assert g['root_index']==z['root_index']==p['root_index']
                root=data['roots'][split][p['root_index']]; h=len(root['history'])
                assert g['search_ply']==z['search_ply']==p['search_ply'] and p['search_ply']>=1
                assert g['history'][h]!=z['history'][h]
                assert p['sign']==(1 if chess.Board(root['fen']).turn else -1)
                assert p['cp_loss']==p['sign']*(g['score_cp']-z['score_cp']) and p['cp_loss']>=15
                margin=p['sign']*(g['strategic_base_cp']-z['strategic_base_cp'])
                maximum=attainable_margin_cp(*[torch.tensor(v) for v in
                    (g['strategic_base_cp'],z['strategic_base_cp'],.25,.25,p['sign'])],250).item()
                assert maximum==p['maximum_signed_margin_cp'] and maximum>0
                assert margin==p['base_margin_cp'] and p['kind']==('repair' if margin<=0 else 'retention')
                endpoints={key(p['good_fen']),key(p['bad_fen'])}
                assert len(endpoints)==2 and not endpoints&pair_endpoints
                pair_endpoints.update(endpoints)
            for r in data['roots'][split]:
                b=reconstruct(r); s=sources[r['game_id']]
                assert b.fen()==r['fen'] and s['history'][:len(r['history'])]==r['history']
                aliases[split].add(key(r['fen']))
            counts[split]={'actual_static_labels':len(rows),'categories':dict(Counter(r['category'] for r in rows)),
                           'source_games':sum(r['split']==split for r in sources.values()),
                           'families':len(families[split]),'sampled_roots':len(data['roots'][split]),
                           'static_pairs':len(pairs[split]),'pair_kinds':dict(Counter(p['kind'] for p in pairs[split]))}
        overlaps={}
        for a,b in (('train','val'),('train','test'),('val','test')):
            overlaps[a+':'+b]={'aliases':len(aliases[a]&aliases[b]),'families':len(families[a]&families[b])}
        assert all(not r['aliases'] and not r['families'] for r in overlaps.values())
        heldout=aliases['val']|aliases['test']
        assert len(retention)==1000 and not {key(r['fen']) for r in retention}&heldout
        for split,rs in roots.items():
            assert len(rs)==24 and len({r['game_id'] for r in rs})==24 and len({r['family'] for r in rs})>=12
            assert len({key(r['fen']) for r in rs})==24
            assert max(Counter(r['family'] for r in rs).values())<=2
            root_counts[split]={'roots':len(rs),'families':len({r['family'] for r in rs}),
                                'colors':dict(Counter('white' if chess.Board(r['fen']).turn else 'black' for r in rs)),
                                'stages':dict(Counter(r['screen_stage'] for r in rs))}
        # Compare the entire instrumented and uninstrumented node-limited tree.
        instrumented=[]
        for r in data['roots']['train'][:6]:
            b=reconstruct(r); workers=[StaticSampler(b,r['game_id']+r['source_ply']),
                                      _Search(b,StrategicEvaluator(PRIORS),None,True,2000)]
            outputs=[]
            for w in workers:
                w.use_lmr=True; w.move_hints[_position_key(b)]=w.fallback(b,list(b.legal_moves),None)
                completed=0; score=None
                for d in (1,2,3):
                    try: value=w.negamax(b,d,-INF,INF,0)
                    except _SearchLimit: break
                    score=value; completed=d
                outputs.append((completed,score,w.nodes,w.qnodes,w.root_candidate,w.static_cache_hits,dict(w.counts)))
            assert outputs[0]==outputs[1] and b.fen()==r['fen']
            instrumented.append({'fen':r['fen'],'same_tree':True,'nodes':outputs[0][2]})
        # Re-query static labels as a target consistency audit. These post-fit
        # values never change selection.
        teacher_checks=[]
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            for split in ('train','val','test'):
                selected=[]
                for category in ('capture','quiet'):
                    selected.extend([r for r in data['rows'][split] if r['category']==category][:6])
                for r in selected:
                    b=reconstruct(r); label=static_evaluation(sf,b)
                    assert label['score_cp']==r['score_cp']
                    teacher_checks.append({'fen':r['fen'],'split':split,'score_cp':r['score_cp'],'requery_matches':True})
            for r in retention[:10]:
                assert static_evaluation(sf,chess.Board(r['fen']))['score_cp']==r['score_cp']
        protocol=json.loads((ART/'protocol.json').read_text())
        assert all(digest(ROOT/p)==h for p,h in protocol['source_hashes'].items())
        assert digest(PREVIOUS)==protocol['previous_sha256'] and digest(SF)==protocol['teacher_sha256']
        exploration=None
        if (ART/'exploratory-protocol.json').exists():
            amendment=json.loads((ART/'exploratory-protocol.json').read_text())
            exploration=json.loads((ART/'exploratory-decision.json').read_text())
            assert all(digest(ROOT/p)==h for p,h in amendment['hashes'].items())
            assert digest(ART/(amendment['candidate']+'.pt'))==amendment['checkpoint_sha256']
            assert digest(ART/'fresh-roots.json')==amendment['fresh_roots_sha256']
            assert exploration['original_validation_gate_passed'] is False
            summary=json.loads((ART/'test-summary.json').read_text())
            assert gate(summary,amendment['candidate'])==exploration['strict_independent_test_passed']
        inference_audit(sorted(ART.glob('epoch-*.pt')),data)
        totals=Counter()
        for r in data['sampling']:
            totals.update(r['stats']['calls'])
        legal_pilot=0
        if (ART/'pilot/report.json').exists():
            pilot=json.loads((ART/'pilot/report.json').read_text()); stream=io.StringIO((ART/'pilot/games.pgn').read_text())
            for record in pilot['games']:
                game=chess.pgn.read_game(stream); assert game and not game.errors
                b=game.end().board()
                assert b.fen()==record['final_fen'] and b.result()==record['result']==game.headers['Result']
                assert [m.uci() for m in game.mainline_moves()]==record['opening_moves']+[m['uci'] for m in record['moves']]
                legal_pilot+=1
            assert chess.pgn.read_game(stream) is None
        result={'counts':counts,'fresh_screen_coverage':root_counts,'cross_split_overlap':overlaps,
                'known_alias_and_source_family_overlap':0,'ordinary_retention_heldout_overlap':0,
                'call_counts':dict(totals),'old_gate_activation_fraction':totals['quiet']/sum(totals.values()),
                'new_gate_activation_fraction':1.,'instrumentation_checks':instrumented,
                'teacher_static_requeries':teacher_checks,'additional_retention_requeries':10,
                'source_and_checkpoint_hashes_match':True,'legal_pilot_games':legal_pilot,
                'exploratory_test':exploration,
                'limits':'Static pairs are not root-move rankings. Sampled leaves within games correlate. Reservoir stratifies unique calls, not natural frequencies. Legacy family provenance incomplete. No online Stockfish in candidate.'}
        write(ART/'audit.json',result); print(json.dumps({k:v for k,v in result.items() if k not in ('teacher_static_requeries','instrumentation_checks')},indent=2))


if __name__=='__main__':
    main()
