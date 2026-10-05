"""Independent frozen confirmation of v26; no fitting or adaptive tuning."""
import hashlib
import io
import json
import random
from dataclasses import asdict
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import opening_board, play_game, summarize
from benchmarks.diagnose_bishop_v25 import target
from engine.evaluation import evaluate
from engine.search import search
from engine.strategic import PRIORS, StrategicEvaluator
from ml.generate_search_data import key
from ml.train_critical_v23 import board
from ml.diagnose_critical_v23 import SF, review
from ml.confirm_fatal_v23 import fatal

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'benchmarks/results/strategy-v27'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def uncertainty(games):
    pairs = {}
    for g in games:
        if g['current_result'] in ('win', 'draw', 'loss'):
            pairs.setdefault(g['pair'], []).append({'win':1., 'draw':.5, 'loss':0.}[g['current_result']])
    values = [sum(v)/2 for v in pairs.values() if len(v)==2]
    rng = random.Random(270027)
    boot = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(10000))
    return {'pairs':len(values), 'score':sum(values)/len(values),
            'interval_95':[boot[250], boot[9749]], 'method':'Opening-pair percentile bootstrap, 10000 resamples, seed 270027'}

def selector(name, b, time_limit, depth_cap):
    model = StrategicEvaluator(PRIORS) if name=='current' else evaluate
    result = search(b, depth=depth_cap, time_limit=time_limit, eval_fn=model, use_lmr=True)
    stats = asdict(result)
    stats.pop('move')
    stats.update(budget_seconds=time_limit, overrun_seconds=max(0, result.elapsed-time_limit))
    return result.move, stats

def screen():
    path = ART / 'regressions.json'
    if path.exists():
        return json.loads(path.read_text())
    cases = [r['root'] for r in json.loads((ROOT/'benchmarks/results/lmr-v24/mistakes.json').read_text())]
    extra = target()
    if not any(r['fen']==extra['fen'] for r in cases):
        cases.append(extra)
    results = []
    with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1, 'Hash':32})
        for i, root in enumerate(cases):
            probes = {}
            teacher = {}
            for budget in (.25,.75):
                probes[str(budget)] = {}
                for name in (('baseline','current') if i%2==0 else ('current','baseline')):
                    move, stats = selector(name, board(root), budget, 64)
                    if move.uci() not in teacher:
                        teacher[move.uci()] = review(sf, board(root), move, 256000)
                    probes[str(budget)][name] = {**stats, 'move':move.uci(), 'review':teacher[move.uci()]}
            results.append({'root':root, 'probes':probes})
            print('Regression position', i+1, '/', len(cases), flush=True)
    summary = {}
    for budget in ('.25','.75'):
        budget = str(float(budget))
        summary[budget] = {}
        for name in ('baseline','current'):
            reviews = [r['probes'][budget][name]['review'] for r in results]
            summary[budget][name] = {'large_errors':sum((r['cp_loss'] or 0)>=150 for r in reviews),
                'losing_transitions':sum(fatal(r) for r in reviews)}
    passed = all(summary[b]['current'][metric]<=summary[b]['baseline'][metric]
                 for b in summary for metric in ('large_errors','losing_transitions'))
    output = {'records':results, 'summary':summary, 'nonregression_passed':passed,
              'limits':'Known development failures, not independent strength evidence; no candidate tuning.'}
    save(path, output)
    return output

def games(label, budget, openings, depth):
    out = ART / label
    out.mkdir(exist_ok=True)
    path = out/'report.json'
    report = json.loads(path.read_text()) if path.exists() else {'status':'running', 'games':[],
        'planned_games':len(openings)*2, 'budget_seconds':budget, 'depth_cap':depth}
    pgns = (out/'games.pgn').read_text() if (out/'games.pgn').exists() else ''
    for i, opening in enumerate(openings, 1):
        for color in (chess.WHITE, chess.BLACK):
            if any(g['pair']==i and g['current_color']==('white' if color else 'black') for g in report['games']):
                continue
            # App confirmation uses the actual app depth cap, eight.
            def choose(name,b,time_limit,depth_cap):
                return selector(name,b,time_limit,depth)
            record, pgn = play_game(opening,color,budget,600,depth,choose,i,'baseline')
            report['games'].append(record)
            pgns += pgn
            report['summary'] = summarize(report['games'])
            report['status'] = 'completed' if len(report['games'])==len(openings)*2 else 'running'
            (out/'games.pgn').write_text(pgns)
            save(path, report)
            s = report['summary']
            print(label, len(report['games']), '/', len(openings)*2, s['wins'],'W',s['draws'],'D',s['losses'],'L',flush=True)
            if s['errors'] or s['unfinished'] or s['interrupted']:
                raise RuntimeError('Incomplete/error game; preserve evidence and investigate')
    u = uncertainty(report['games'])
    save(out/'uncertainty.json', u)
    # Audit legal moves, recorded histories and terminal results.
    stream = io.StringIO(pgns)
    for record in report['games']:
        g = chess.pgn.read_game(stream)
        assert g and not g.errors
        b = g.board()
        moves = []
        for move in g.mainline_moves():
            assert move in b.legal_moves
            b.push(move)
            moves.append(move.uci())
        assert moves==record['opening_moves']+[m['uci'] for m in record['moves']]
        assert b.fen()==record['final_fen']
        assert b.outcome(claim_draw=False).result()==record['result']
    assert chess.pgn.read_game(stream) is None
    return report['summary'], u

def main():
    ART.mkdir(parents=True,exist_ok=True)
    paths = [ROOT/'engine/strategic.py', ROOT/'engine/search.py', ROOT/'engine/evaluation.py', Path(__file__)]
    protocol = {'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths},
        'weights':list(PRIORS), 'independent':{'games':200,'seconds':.25,'depth_cap':64},
        'app_budget':{'games':40,'seconds':.75,'depth_cap':8},
        'promotion':'Independent score >=60%, pair-bootstrap lower bound >50%, app-budget score >50%, known-failure nonregression, all games legal and complete, tests pass. No tuning after results.',
        'timing':'Exclusive CPU; equal nominal search budgets; evaluator setup outside timer; no NN or book after prefix.',
        'openings_hashes':{name:digest(ROOT/f'benchmarks/openings-strategy-v27-{name}.json') for name in ('independent','app')}}
    frozen = ART/'protocol.json'
    if frozen.exists():
        assert json.loads(frozen.read_text())==protocol, 'Frozen source/protocol changed'
    else:
        save(frozen,protocol)
    forbidden = set()
    for s in ('training-labels','train-roots','val-roots','test-roots'):
        forbidden.update(key(r['fen']) for r in json.loads((ROOT/f'benchmarks/results/strategy-v26/{s}.json').read_text()))
    openings = {name:json.loads((ROOT/f'benchmarks/openings-strategy-v27-{name}.json').read_text()) for name in ('independent','app')}
    assert not any(key(opening_board(o).fen()) in forbidden for rows in openings.values() for o in rows)
    with exclusive_cpu('v27 independent confirmation and app-budget validation'):
        regression = screen()
        independent, u = games('independent',.25,openings['independent'],64)
        app, app_u = games('app',.75,openings['app'],8)
    strength = independent['score_fraction_completed']>=.6 and u['interval_95'][0]>.5
    app_pass = app['score_fraction_completed']>.5
    save(ART/'decision.json', {'strength_passed':strength, 'app_budget_passed':app_pass,
        'regressions_passed':regression['nonregression_passed'],
        'eligible_after_tests':strength and app_pass and regression['nonregression_passed'],
        'app_promoted':False, 'independent':independent,'app':app,'independent_uncertainty':u,'app_uncertainty':app_u})
    print('All confirmation games and audits complete.',flush=True)

if __name__=='__main__':
    main()
