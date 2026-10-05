"""Independent provenance, feature-use and optional game checks after v31."""
import io
import json
from collections import Counter
import chess
import chess.pgn
import torch
from benchmarks.cpu_lock import exclusive_cpu
from ml.run_threat_v31 import ART, ROOT, PREVIOUS, all_known_aliases, inference_audit
from ml.run_balanced_v19 import write, digest
from ml.generate_search_data import key
from ml.run_search_correction_v29 import family
from ml.threat_hybrid import ThreatHybrid
from ml.strategic_hybrid import StrategicHybrid
from engine.strategic import PRIORS, StrategicEvaluator


def main():
    with exclusive_cpu('v31 final provenance and game audit'):
        torch.set_num_threads(1)
        roots = json.loads((ART/'fresh-roots.json').read_text())
        source = json.loads((ART/'source-collection.json').read_text())
        known = all_known_aliases()
        old = json.loads((ROOT/'ml/artifacts/ranking-v30/collection.json').read_text())
        known_families = {r['family'] for r in old['sources']}
        known_families.update(r['family'] for r in json.loads(
            (ROOT/'ml/artifacts/search-correction-v29/source-games.json').read_text()))
        sources = {r['game_id']:r for r in source['sources']}
        for r in sources.values():
            b = chess.Board()
            for u in r['history']:
                b.push_uci(u)
            assert b.fen() == r['final_fen'] and family(r['history']) == r['family']
        aliases = {}; families = {}; coverage = {}
        for split, rs in roots.items():
            assert len(rs) == 24 and len({r['game_id'] for r in rs}) == 24
            aliases[split] = {key(r['fen']) for r in rs}
            families[split] = {r['family'] for r in rs}
            assert not aliases[split] & known and not families[split] & known_families
            assert max(Counter(r['family'] for r in rs).values()) <= 2
            assert len(families[split]) >= 12
            colors = Counter(); stages = Counter()
            for r in rs:
                b = chess.Board(r['initial_fen'])
                for u in r['history']:
                    b.push_uci(u)
                assert b.fen() == r['fen'] and not b.is_game_over()
                s = sources[r['game_id']]
                assert s['split'] == split and s['family'] == r['family']
                assert s['history'][:len(r['history'])] == r['history']
                assert r['teacher_white_cp'] is not None and abs(r['teacher_white_cp']) <= 800
                colors['white' if b.turn else 'black'] += 1; stages[r['screen_stage']] += 1
            assert colors['white'] and colors['black']
            coverage[split] = {'roots':len(rs),'games':24,'families':len(families[split]),
                               'colors':dict(colors),'stages':dict(stages)}
        assert not aliases['val'] & aliases['test'] and not families['val'] & families['test']
        protocol = json.loads((ART/'protocol.json').read_text())
        assert all(digest(ROOT/p) == h for p,h in protocol['hashes'].items())
        assert digest(PREVIOUS) == protocol['previous_checkpoint_sha256']
        paths = sorted(ART.glob('epoch-*.pt')); inference_audit(paths, roots)
        features = {}
        for p in paths:
            weights = torch.load(p,map_location='cpu',weights_only=True)['state_dict']['net.0.weight'][:,892:]
            norm = weights.norm().item(); assert norm > 0
            features[p.name] = {'extra_feature_weights_l2':norm}
        # Exposed v30 endpoints describe retention, never select this round.
        endpoints = {}
        for split in ('train','val'):
            endpoints[split] = {}
            models = {'heuristic':StrategicEvaluator(PRIORS),'previous_nn':StrategicHybrid(PREVIOUS,.25),
                      **{p.stem:ThreatHybrid(p) for p in paths}}
            for name,model in models.items():
                counts = Counter()
                for pair in old['pairs'][split]:
                    gap = pair['sign']*(model(chess.Board(pair['good_fen']))-model(chess.Board(pair['bad_fen'])))
                    counts['pairs'] += 1; counts['correct'] += int(gap > 0)
                    counts[pair['kind']+'_pairs'] += 1; counts[pair['kind']+'_correct'] += int(gap > 0)
                    counts['retention_regressions'] += int(pair['kind']=='retention' and gap <= 0)
                endpoints[split][name] = dict(counts)
        write(ART/'exposed-endpoint-metrics.json',{'selection_uses_these_metrics':False,'splits':endpoints})
        pilot_count = 0
        if (ART/'pilot/report.json').exists():
            pilot = json.loads((ART/'pilot/report.json').read_text())
            pgns = io.StringIO((ART/'pilot/games.pgn').read_text())
            for record in pilot['games']:
                game = chess.pgn.read_game(pgns); assert game and not game.errors
                b = game.end().board()
                assert b.fen() == record['final_fen'] and b.result() == record['result'] == game.headers['Result']
                assert [m.uci() for m in game.mainline_moves()] == record['opening_moves'] + [m['uci'] for m in record['moves']]
                pilot_count += 1
            assert chess.pgn.read_game(pgns) is None
        gates = json.loads((ART/'gate-audit.json').read_text())
        assert all(r['same_move_as_uninstrumented'] for r in gates['roots'])
        result = {'fresh_coverage':coverage,'legal_source_trajectories':len(sources),
                  'fresh_cross_split_position_and_family_overlap':0,
                  'known_alias_and_v29_v30_source_family_overlap':0,
                  'frozen_hashes_match':True,'gate_audit_preserves_moves':True,
                  'learned_extra_feature_weights':features,'legal_pilot_games':pilot_count,
                  'limitations':'Legacy training families incompletely documented. Representation and retention changes combined; no causal feature ablation. Exposed v30 rankings are descriptive only.'}
        write(ART/'audit.json', result); print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
