"""Audit actual search gates, expand representation, fit once, then fresh gates."""
import copy
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import chess
import chess.engine
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from benchmarks.match import opening_board, play_game, summarize
from engine.search import search, _Search, INF, _position_key
from engine.strategic import PRIORS, StrategicEvaluator
from ml import run_strategic_v28 as common
from ml.run_balanced_v19 import write, digest, SF
from ml.run_search_v22 import exact
from ml.run_search_correction_v29 import family, quiet
from ml.run_ranking_v30 import reserved_aliases, phase
from ml.generate_search_data import key
from ml.diagnose_critical_v23 import forced
from ml.strategic_hybrid import StrategicHybrid
from ml.threat_model import (THREAT_INPUT_SIZE, THREAT_VERSION, ThreatNet,
                            board_to_threat_array, threat_features)
from ml.threat_hybrid import ThreatHybrid
from ml.model import SCORE_SCALE
from ml.train import rounded_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'ml/artifacts/threat-v31'
PREVIOUS = ROOT / 'ml/artifacts/ranking-v30/epoch-24.pt'
WEIGHT = .25


class GateAuditSearch(_Search):
    def __init__(self, board, evaluator):
        super().__init__(board, evaluator, None, True)
        self.use_lmr = True
        self.gates = Counter()
        self.unique = {}

    def static(self, board):
        active = quiet(board)
        reason = 'active' if active else ('check' if board.is_check() else 'capture')
        self.gates[reason] += 1
        self.unique[board.fen()] = reason
        return super().static(board)


def audit_gate(sf):
    path = ART / 'gate-audit.json'
    if path.exists():
        return json.loads(path.read_text())
    records = json.loads((ROOT / 'ml/artifacts/ranking-v30/test.json').read_text())
    base = StrategicEvaluator(PRIORS)
    rows = []
    for i, r in enumerate(records):
        b = reconstruct(r['root'])
        worker = GateAuditSearch(b, base)
        moves = list(b.legal_moves)
        hint = worker.fallback(b, moves, None)
        worker.move_hints[_position_key(b)] = hint
        for depth in (1, 2, 3):
            worker.negamax(b, depth, -INF, INF, 0)
        reference = search(b.copy(stack=True), depth=3, eval_fn=base, use_lmr=True)
        assert worker.root_candidate == reference.move and b.fen() == r['root']['fen']
        rows.append({'fen':b.fen(), 'calls':dict(worker.gates),
                     'unique':dict(Counter(worker.unique.values())),
                     'nodes':worker.nodes, 'qnodes':worker.qnodes,
                     'same_move_as_uninstrumented':True})
        if (i + 1) % 8 == 0:
            print('Gate audit', i + 1, '/', len(records), flush=True)
    # Previously exposed failures are diagnostic only, never fresh test evidence.
    failures = [r for r in records if
                (r['probes']['250ms']['heuristic']['review']['cp_loss'] or 0) >= 150]
    traces = []
    for r in failures[:8]:
        review = r['probes']['250ms']['heuristic']['review']
        root = reconstruct(r['root'])
        branches = {}
        for label, move in [('heuristic', r['probes']['250ms']['heuristic']['move']),
                            ('teacher', review['best_move'])]:
            t = forced(root.copy(stack=True), chess.Move.from_uci(move), base, depth=3)
            leaf = t['leaf']
            if leaf:
                b = chess.Board(root.root().fen())
                for uci in leaf['history']:
                    b.push_uci(uci)
                assert b.fen() == leaf['fen']
                info = exact(sf, b, nodes=64000)
                t['gate_active'] = quiet(b)
                t['teacher_white_cp'] = info['score'].white().score()
                t['has_legal_capture'] = next(b.generate_legal_captures(), None) is not None
                t['pinned_piece_count'] = sum(b.is_pinned(c, s) for c in chess.COLORS
                                            for s in chess.scan_forward(b.occupied_co[c]))
            branches[label] = t
        gap = branches['heuristic']['root_score_cp'] - branches['teacher']['root_score_cp']
        leaves_have_capacity = any(t.get('gate_active', False) for t in branches.values())
        traces.append({'root':r['root'], 'previous_timed_review':review, 'branches':branches,
                       'heuristic_preference_gap_cp':gap,
                       'global_bounded_range_permits_reversal':gap <= 126,
                       'traced_endpoints_have_capacity':leaves_have_capacity})
    totals = Counter()
    unique_totals = Counter()
    for r in rows:
        totals.update(r['calls']); unique_totals.update(r['unique'])
    result = {'roots':rows, 'totals':dict(totals), 'unique_totals_per_root':dict(unique_totals),
              'active_fraction_calls':totals['active'] / sum(totals.values()),
              'critical_branch_traces':traces,
              'meaning':'Actual static calls, including cache hits, in baseline depth3 iterative LMR search. Instrumentation preserves moves. All are quiescence stand-pat evaluations; in-check evasions do not stand pat. Full-window forced branch traces disable TT/LMR and are a different tree; traced endpoints are diagnostics, not root repair proofs. Exposed v30 test roots are not new validation or training.'}
    write(path, result)
    return result


def all_known_aliases():
    aliases = reserved_aliases()
    data = json.loads((ROOT / 'ml/artifacts/ranking-v30/collection.json').read_text())
    for ps in data['pairs'].values():
        aliases.update(key(p[f]) for p in ps for f in ('root_fen','good_fen','bad_fen'))
    for rs in data['roots'].values():
        aliases.update(key(r['fen']) for r in rs)
    return aliases


def fresh_roots(sf):
    path = ART / 'fresh-roots.json'
    if path.exists():
        return json.loads(path.read_text())
    old = json.loads((ROOT / 'ml/artifacts/ranking-v30/collection.json').read_text())
    known_families = {r['family'] for r in old['sources']}
    known_families.update(r['family'] for r in json.loads(
        (ROOT / 'ml/artifacts/search-correction-v29/source-games.json').read_text()))
    groups = defaultdict(list)
    for o in json.loads((ROOT / 'benchmarks/openings-threat-v31-data.json').read_text()):
        f = family(o['moves'])
        if f not in known_families:
            groups[f].append({**o, 'family':f})
    assigned = {'val':[], 'test':[]}
    for f, os in sorted(groups.items(), key=lambda v:(-len(v[1]), v[0])):
        assigned[min(assigned, key=lambda s:len(assigned[s]))] += os
    rng = random.Random(310032)
    for os in assigned.values():
        rng.shuffle(os)
    write(ART / 'source-allocation.json', assigned)
    checkpoint = ART / 'source-collection.json'
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {'sources':[], 'pools':{'val':[], 'test':[]}}
    forbidden = all_known_aliases()
    owners = {key(r['fen']):s for s,rs in state['pools'].items() for r in rs}
    for split, os in assigned.items():
        for gi, o in enumerate(os):
            gid = 3100000 + (10000 if split == 'test' else 0) + gi
            if any(s['game_id'] == gid for s in state['sources']):
                continue
            b = opening_board(o)
            for ply in range(120):
                if b.is_game_over():
                    break
                if ply in (16 + gi % 2, 48 + gi % 2, 80 + gi % 2, 112 + gi % 2):
                    alias = key(b.fen())
                    if alias not in forbidden and alias not in owners:
                        info = exact(sf, b, nodes=64000)
                        cp = info['score'].white().score()
                        if cp is not None and abs(cp) <= 800:
                            r = {'fen':b.fen(), 'initial_fen':b.root().fen(),
                                 'history':[m.uci() for m in b.move_stack], 'opening':o,
                                 'game_id':gid, 'family':o['family'], 'split':split,
                                 'screen_stage':phase(b), 'source_ply':ply,
                                 'teacher_white_cp':cp, 'screen_teacher_nodes':64000}
                            state['pools'][split].append(r); owners[alias] = split
                b.push(exact(sf, b, nodes=3000)['pv'][0])
            state['sources'].append({'game_id':gid, 'split':split, 'family':o['family'],
                                    'opening':o, 'history':[m.uci() for m in b.move_stack],
                                    'final_fen':b.fen(), 'truncated':not b.is_game_over()})
            write(checkpoint, state)
            print('Fresh source', split, gi + 1, '/', len(os), 'pool', len(state['pools'][split]), flush=True)
    result = {}
    for split, pool in state['pools'].items():
        rng.shuffle(pool); stages = Counter(); colors = Counter(); families = Counter(); games = set(); chosen = []
        while len(chosen) < 24:
            candidates = [r for r in pool if r['game_id'] not in games and families[r['family']] < 2]
            if not candidates:
                raise RuntimeError('Fresh source coverage too small; preserve before fitting')
            r = min(candidates, key=lambda r:(families[r['family']], colors[chess.Board(r['fen']).turn], stages[r['screen_stage']]))
            chosen.append(r); games.add(r['game_id']); families[r['family']] += 1
            colors[chess.Board(r['fen']).turn] += 1; stages[r['screen_stage']] += 1
        assert colors[True] and colors[False] and len(families) >= 12
        result[split] = chosen
    assert not {r['family'] for r in result['val']} & {r['family'] for r in result['test']}
    write(path, result)
    return result


def tensor_rows(rs):
    base = StrategicEvaluator(PRIORS)
    return (torch.stack([torch.from_numpy(board_to_threat_array(chess.Board(r['fen']))) for r in rs]),
            torch.tensor([base(chess.Board(r['fen'])) for r in rs], dtype=torch.float32),
            torch.tensor([r['score_cp'] for r in rs], dtype=torch.float32))


def pair_tensors(ps):
    base = StrategicEvaluator(PRIORS); arrays = [[] for _ in range(8)]
    for p in ps:
        g, z = chess.Board(p['good_fen']), chess.Board(p['bad_fen'])
        vals = (torch.from_numpy(board_to_threat_array(g)), torch.from_numpy(board_to_threat_array(z)),
                base(g), base(z), float(quiet(g)), float(quiet(z)), p['sign'], p['cp_loss'])
        for a,v in zip(arrays, vals):
            a.append(v)
    return tuple(torch.stack(a) if i < 2 else torch.tensor(a,dtype=torch.float32) for i,a in enumerate(arrays))


def metadata(state):
    return {'version':THREAT_VERSION, 'input_size':THREAT_INPUT_SIZE, 'score_scale':SCORE_SCALE,
            'baseline_id':'strategic-v26', 'baseline_weights':list(PRIORS),
            'training_correction_weight':WEIGHT, 'training_quiet_only':True,
            'hidden_sizes':[64,32], 'correction_limit_cp':250, 'state_dict':state}


def train_once(roots):
    torch.manual_seed(310033)
    data = json.loads((ROOT / 'ml/artifacts/ranking-v30/collection.json').read_text())
    retained = json.loads((ROOT / 'ml/artifacts/ranking-v30/retained-score-rows.json').read_text())
    heldout = {key(r['fen']) for rs in roots.values() for r in rs}
    rows = retained + data['rows']['train'] * 3
    assert not {key(r['fen']) for r in rows} & heldout
    assert not {key(p[f]) for p in data['pairs']['train'] for f in ('root_fen','good_fen','bad_fen')} & heldout
    scores = tensor_rows(rows); ranks = pair_tensors(data['pairs']['train'])
    importance = torch.tensor([3. if p['kind'] == 'repair' else 1. for p in data['pairs']['train']])
    model = ThreatNet()
    old = torch.load(PREVIOUS, map_location='cpu', weights_only=True)['state_dict']
    state = model.state_dict()
    for k in state:
        if k == 'net.0.weight':
            state[k].zero_(); state[k][:,:892] = old[k]
        else:
            state[k] = old[k].clone()
    model.load_state_dict(state)
    torch.save(metadata(copy.deepcopy(state)), ART / 'initial-expanded.pt')
    loader = DataLoader(TensorDataset(*scores), batch_size=128, shuffle=True)
    rankloader = DataLoader(TensorDataset(*ranks,importance), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=.0002)
    history = []; paths = []
    for epoch in range(1,25):
        model.train()
        for x,b,y in loader:
            opt.zero_grad(); delta = WEIGHT * model(x)
            pred = rounded_score(b,delta,True)
            loss = 4*nn.functional.mse_loss(torch.sigmoid(pred/400),torch.sigmoid(y/400)) + .02*delta.square().mean()
            loss.backward(); opt.step()
        for g,z,gb,zb,gf,zf,sign,cp,importance in rankloader:
            opt.zero_grad()
            gap = sign*(rounded_score(gb,WEIGHT*gf*model(g),True)-rounded_score(zb,WEIGHT*zf*model(z),True))
            ref = sign*(gb-zb)
            rankloss = nn.functional.binary_cross_entropy_with_logits(gap/100,torch.sigmoid(cp.clamp(max=500)/100),reduction='none')
            protection = torch.where(ref>0, torch.relu(ref.clamp(max=20)-gap)/100, torch.zeros_like(gap))
            ((rankloss*importance).mean()+10*protection.mean()).backward(); opt.step()
        if epoch in (8,16,24):
            path = ART / f'epoch-{epoch}.pt'; torch.save(metadata(copy.deepcopy(model.state_dict())),path); paths.append(path)
            model.eval()
            with torch.inference_mode():
                g,z,gb,zb,gf,zf,sign,cp = ranks
                gap = sign*(rounded_score(gb,WEIGHT*gf*model(g))-rounded_score(zb,WEIGHT*zf*model(z)))
                repairs = importance == 3
                row = {'epoch':epoch, 'correct':int((gap>0).sum()), 'pairs':len(gap),
                       'repair_correct':int(((gap>0)&repairs).sum()),
                       'retention_regressions':int(((gap<=0)&~repairs).sum())}
                history.append(row); print('Training',row,flush=True)
    write(ART/'training.json',{'history':history,'score_samples':len(rows),'ranking_pairs':len(importance),
                             'single_run':True,'warm_start_extra_weights_zero':True,
                             'retention_protection_coefficient':10,'protected_margin_cap_cp':20})
    return paths


def gate(summary, name):
    return common.qualifies(summary,name) and all(rs[name]['comparable']>=20 for rs in summary.values())


def inference_audit(paths, roots):
    positions = [r['fen'] for rs in roots.values() for r in rs]
    data = json.loads((ROOT/'ml/artifacts/ranking-v30/collection.json').read_text())
    positions += [r['fen'] for r in data['rows']['train'][:52]]
    results = {}
    for path in paths:
        adapter = ThreatHybrid(path); maximum = 0
        for fen in positions:
            b = chess.Board(fen); factor = WEIGHT if quiet(b) else 0
            with torch.inference_mode():
                expected = round(adapter.baseline(b)+factor*adapter.model(torch.from_numpy(board_to_threat_array(b))).item()*SCORE_SCALE)
            actual = adapter.evaluate_position(b)
            maximum = max(maximum,abs(actual-expected))
            assert abs(actual-expected)<=1 and actual == -adapter.evaluate_position(b.mirror()) and b.fen()==fen
        results[path.name] = {'positions':len(positions),'max_rounding_difference_cp':maximum,'color_symmetry':True}
    # Extra-feature weights start at zero, preserving the old network exactly.
    expanded = ThreatHybrid(ART/'initial-expanded.pt'); old = StrategicHybrid(PREVIOUS,WEIGHT)
    assert all(expanded.evaluate_position(chess.Board(f)) == old.evaluate_position(chess.Board(f)) for f in positions)
    write(ART/'inference-audit.json',{'checkpoints':results,'expanded_initial_matches_previous':True})


def pilot(path):
    import subprocess, sys
    output = ROOT/'benchmarks/openings-threat-v31-pilot.json'
    subprocess.run([sys.executable,'-m','benchmarks.generate_strength_openings','--output',str(output),'--pairs','10','--seed','310034'],check=True)
    candidate = ThreatHybrid(path); baseline = StrategicEvaluator(PRIORS)
    def selector(name,b,time_limit,depth_cap):
        p = search(b,depth=64,time_limit=time_limit,eval_fn=candidate if name=='current' else baseline,use_lmr=True)
        stats = asdict(p); stats.pop('move'); return p.move, stats
    games = []; pgns = ''
    for i,o in enumerate(json.loads(output.read_text()),1):
        for color in chess.COLORS:
            record,pgn = play_game(o,color,.25,600,64,selector,i,'new_heuristic')
            games.append(record);pgns+=pgn; summary=summarize(games)
            write(ART/'pilot/report.json',{'games':games,'summary':summary,'status':'running','planned_games':20})
            (ART/'pilot/games.pgn').write_text(pgns); print('Pilot',len(games),summary,flush=True)
            if summary['errors'] or summary['unfinished'] or summary['interrupted']:
                raise RuntimeError('Incomplete pilot; preserve records')
    write(ART/'pilot/report.json',{'games':games,'summary':summary,'status':'completed','planned_games':20})


def main():
    ART.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    protocol = {'previous_checkpoint_sha256':digest(PREVIOUS), 'architecture':'1420 inputs, unchanged64/32 hidden; pin-aware square attacker/defender counts, absolute pins, cheaper attackers; king-ring threats. Extra input weights initially zero.',
                'gate':'Unchanged strict quiet25% with raw250cp bound and exact rounded hybrid; audited on exposed v30 test only, not used for fitting.',
                'training':'One24epoch run, checkpoints8/16/24 only; v30train300pairs and3800score samples; no exposed v30val/test labels enter fitting; lr.0002; repair importance3; retention protection10 vs4, cap20 vs10; correction regularization.02 vs.01.',
                'fresh_screen':'320newbookstarts seed310031; exclude known v29/v30sourcefamilies and all known label/root aliases. Fresh teacher selfplay3000nodes max120plies, snapshots16/48/80/112withbalancedcoloroffset; teacher64k nonmate absWhitecp<=800. Whole6plyfamilies splitval/test;24roots each, one/game,max2/family, >=12families. Freeze before fit; teacher-only coverage selection.',
                'selection':'Fresh depth3 and250ms/depth64 searched validation; select eligible checkpoint with lowest timed mean, common>=20, nonworse depth mean, strictly better timed mean, noextra>=150cp errors, losing transitions ormates. Selected checkpoint tested once, never tuned on test. Pilot20 only aftertestpass, noextension/appadoptionautomatic.',
                'limits':'Pin-aware pseudo attacks are not full capture legality or SEE. Gate traces arediagnostic, notrepairproof. Legacyfamilyprovenanceincomplete. Morefeaturesandstrongerretentionarecombined; notanisolatedfeatureablation.',
                'hashes':{p:digest(ROOT/p) for p in ('engine/search.py','engine/strategic.py','ml/run_threat_v31.py','ml/threat_model.py','ml/threat_hybrid.py','benchmarks/openings-threat-v31-data.json')}}
    path = ART/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==protocol, 'Frozen protocol changed'
    else:
        write(path,protocol)
    with exclusive_cpu('v31 gate audit representation training and fresh gates'):
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            audit_gate(sf)
            roots = fresh_roots(sf)
            paths = train_once(roots)
            inference_audit(paths,roots)
            common.ART = ART
            models = {'heuristic':StrategicEvaluator(PRIORS),'previous_nn':StrategicHybrid(PREVIOUS,WEIGHT),
                      **{p.stem:ThreatHybrid(p) for p in paths}}
            development = common.screen(sf,roots['val'],models,'development')
            eligible = [p.stem for p in paths if gate(development,p.stem)]
            selected = min(eligible,key=lambda n:development['250ms'][n]['mean_regret']) if eligible else None
            write(ART/'selection.json',{'selected':selected,'test_used_for_selection':False})
            if not selected:
                write(ART/'decision.json',{'qualifies_for_games':False,'reason':'No snapshot passed searched validation','app_unchanged':True});return
            test = common.screen(sf,roots['test'],{n:models[n] for n in ('heuristic',selected)},'test')
            passed = gate(test,selected)
            write(ART/'decision.json',{'selected':selected,'qualifies_for_games':passed,'app_unchanged':True})
        if passed:
            pilot(ART/f'{selected}.pt')


if __name__ == '__main__':
    main()
