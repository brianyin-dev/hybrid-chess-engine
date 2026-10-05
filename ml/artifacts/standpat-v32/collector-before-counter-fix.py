"""Actual static-call supervision and a matched non-check gate experiment."""
import copy
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import chess
import chess.engine
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from benchmarks.match import opening_board, play_game, summarize
from engine.search import search, _Search, _SearchLimit, INF, _position_key
from engine.strategic import PRIORS, StrategicEvaluator
from ml import run_strategic_v28 as common
from ml.run_balanced_v19 import write, digest, SF
from ml.run_search_v22 import exact
from ml.run_search_correction_v29 import family
from ml.run_ranking_v30 import reserved_aliases, phase
from ml.generate_search_data import key
from ml.threat_model import THREAT_INPUT_SIZE, THREAT_VERSION, ThreatNet, board_to_threat_array
from ml.threat_hybrid import ThreatHybrid
from ml.standpat_hybrid import StandPatHybrid, GATE_ID, gate_active
from ml.static_teacher import static_evaluation
from ml.model import SCORE_SCALE
from ml.train import rounded_score, attainable_margin_cp

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'ml/artifacts/standpat-v32'
PREVIOUS = ROOT/'ml/artifacts/threat-v31/epoch-16.pt'
WEIGHT = .25


def known_provenance():
    aliases = reserved_aliases(); families = set()
    old = json.loads((ROOT/'ml/artifacts/ranking-v30/collection.json').read_text())
    families.update(r['family'] for r in old['sources'])
    for ps in old['pairs'].values():
        aliases.update(key(p[f]) for p in ps for f in ('root_fen','good_fen','bad_fen'))
    for rs in old['roots'].values():
        aliases.update(key(r['fen']) for r in rs)
    families.update(r['family'] for r in json.loads((ROOT/'ml/artifacts/search-correction-v29/source-games.json').read_text()))
    prior = json.loads((ROOT/'ml/artifacts/threat-v31/source-collection.json').read_text())
    families.update(r['family'] for r in prior['sources'])
    for rs in prior['pools'].values():
        aliases.update(key(r['fen']) for r in rs)
    return aliases,families


class StaticSampler(_Search):
    """Reservoir of unique actual calls, stratified capture/quiet, no score filter."""
    def __init__(self, board, seed):
        super().__init__(board,StrategicEvaluator(PRIORS),None,True,2000)
        self.use_lmr = True; self.rng = random.Random(seed)
        self.calls = Counter(); self.unique = set(); self.counts = Counter()
        self.samples = {'quiet':[], 'capture':[]}

    def static(self, board):
        assert gate_active(board), 'Quiescence must not stand pat in check'
        category = 'capture' if next(board.generate_legal_captures(),None) is not None else 'quiet'
        self.calls[category] += 1
        alias = key(board.fen())
        if alias not in self.unique:
            self.unique.add(alias); self.counts[category] += 1
            index = self.rng.randrange(self.counts[category])
            if len(self.samples[category]) < 16 or index < 16:
                row = {'fen':board.fen(),'initial_fen':board.root().fen(),
                       'history':[m.uci() for m in board.move_stack],'category':category}
                if len(self.samples[category]) < 16:
                    self.samples[category].append(row)
                else:
                    self.samples[category][index] = row
        return super().static(board)


def sample_root(root, seed):
    b = reconstruct(root); worker = StaticSampler(b,seed)
    worker.move_hints[_position_key(b)] = worker.fallback(b,list(b.legal_moves),None)
    completed = 0
    for depth in (1,2,3):
        try:
            worker.negamax(b,depth,-INF,INF,0)
        except _SearchLimit:
            break
        completed = depth
    assert b.fen() == root['fen'] and len(b.move_stack) == len(root['history'])
    rows = [r for rs in worker.samples.values() for r in rs]
    for r in rows:
        r.update(game_id=root['game_id'],family=root['family'],split=root['split'],
                 root_fen=root['fen'],root_history_len=len(root['history']),
                 search_ply=len(r['history'])-len(root['history']),
                 evaluated_by='heuristic_actual_static_call')
    return rows,{'calls':dict(worker.calls),'unique_calls':dict(worker.counts),
                 'sampled':len(rows),'completed_depth':completed,'nodes':worker.nodes}


def allocation():
    path = ART/'allocation.json'
    if path.exists():
        return json.loads(path.read_text())
    _,known_families = known_provenance(); groups = defaultdict(list)
    for o in json.loads((ROOT/'benchmarks/openings-standpat-v32-data.json').read_text()):
        f = family(o['moves'])
        if f not in known_families:
            groups[f].append({**o,'family':f})
    assigned = {s:[] for s in ('train','val','test')}; ratio = {'train':2,'val':1,'test':1}
    for f,os in sorted(groups.items(),key=lambda v:(-len(v[1]),v[0])):
        assigned[min(assigned,key=lambda s:len(assigned[s])/ratio[s])] += os
    rng = random.Random(320033)
    for os in assigned.values():
        rng.shuffle(os)
    assert min(len(assigned[s]) for s in ('val','test')) >= 28
    write(path,assigned); return assigned


def collect(sf):
    path = ART/'collection.json'
    data = json.loads(path.read_text()) if path.exists() else {
        'sources':[],'roots':{s:[] for s in ('train','val','test')},
        'rows':{s:[] for s in ('train','val','test')},'sampling':[], 'complete':False}
    if data['complete']:
        return data
    forbidden,_ = known_provenance()
    owners = {key(r['fen']):s for s,rs in data['rows'].items() for r in rs}
    root_owners = {key(r['fen']):s for s,rs in data['roots'].items() for r in rs}
    base = StrategicEvaluator(PRIORS)
    for split,os in allocation().items():
        for gi,o in enumerate(os):
            gid = 3200000 + 10000*list(data['rows']).index(split) + gi
            if any(r['game_id'] == gid for r in data['sources']):
                continue
            b = opening_board(o); source_policy = ('stockfish_selfplay','heuristic_selfplay','heuristic_vs_stockfish')[gi%3]
            local_roots = []; local_rows = []; local_sampling = []
            for ply in range(120):
                if b.is_game_over():
                    break
                if ply in (16+gi%2,48+gi%2,80+gi%2,112+gi%2):
                    alias = key(b.fen())
                    info = exact(sf,b,nodes=64000); cp = info['score'].white().score()
                    if cp is not None and abs(cp)<=800 and alias not in forbidden and root_owners.get(alias,split)==split and owners.get(alias,split)==split:
                        root = {'fen':b.fen(),'initial_fen':b.root().fen(),
                                'history':[m.uci() for m in b.move_stack],'game_id':gid,
                                'family':o['family'],'opening':o,'split':split,
                                'source_ply':ply,'screen_stage':phase(b),'teacher_root_white_cp':cp}
                        leaves,stats = sample_root(root,gid+ply)
                        for row in leaves:
                            leaf_alias = key(row['fen'])
                            if leaf_alias in forbidden or leaf_alias in owners or root_owners.get(leaf_alias,split)!=split:
                                continue
                            board = reconstruct(row)
                            assert gate_active(board) and not board.is_game_over(claim_draw=False)
                            row.update(static_evaluation(sf,board)); row['strategic_base_cp'] = base(board)
                            row['root_index'] = len(data['roots'][split])+len(local_roots)
                            local_rows.append(row); owners[leaf_alias] = split
                        local_roots.append(root); root_owners[alias] = split
                        local_sampling.append({'root':root,'stats':stats})
                ours = source_policy=='heuristic_selfplay' or source_policy=='heuristic_vs_stockfish' and b.turn==(gi%2==0)
                move = search(b,depth=64,node_limit=1500,eval_fn=base,use_lmr=True).move if ours else exact(sf,b,nodes=3000)['pv'][0]
                b.push(move)
            data['sources'].append({'game_id':gid,'split':split,'family':o['family'],'opening':o,
                                    'policy':source_policy,'history':[m.uci() for m in b.move_stack],
                                    'final_fen':b.fen(),'truncated':not b.is_game_over()})
            data['roots'][split] += local_roots; data['rows'][split] += local_rows; data['sampling'] += local_sampling
            write(path,data)
            print('Collect',split,gi+1,'/',len(os),'roots',len(data['roots'][split]),'static labels',len(data['rows'][split]),flush=True)
    data['complete'] = True; write(path,data); return data


def freeze_screens(data):
    path = ART/'fresh-roots.json'
    if path.exists():
        return json.loads(path.read_text())
    result = {}; rng = random.Random(320034)
    for split in ('val','test'):
        pool = copy.deepcopy(data['roots'][split]); rng.shuffle(pool)
        games = set(); families = Counter(); stages = Counter(); colors = Counter(); chosen = []
        while len(chosen)<24:
            choices = [r for r in pool if r['game_id'] not in games and families[r['family']]<2]
            if not choices:
                raise RuntimeError('Insufficient fresh screen coverage; preserve data before fitting')
            r = min(choices,key=lambda r:(families[r['family']],colors[chess.Board(r['fen']).turn],stages[r['screen_stage']]))
            chosen.append(r); games.add(r['game_id']); families[r['family']]+=1
            colors[chess.Board(r['fen']).turn]+=1; stages[r['screen_stage']]+=1
        assert len(families)>=12 and colors[True] and colors[False]
        result[split] = chosen
    assert not {r['family'] for r in result['val']} & {r['family'] for r in result['test']}
    write(path,result); return result


def build_pairs(data):
    path = ART/'pairs.json'
    if path.exists():
        return json.loads(path.read_text())
    output = {}; rng = random.Random(320035)
    for split,rows in data['rows'].items():
        groups = defaultdict(list)
        for r in rows:
            groups[r['root_index']].append(r)
        pairs = []; used = set()
        for ri,leaves in groups.items():
            root = data['roots'][split][ri]; sign = 1 if chess.Board(root['fen']).turn else -1
            options = {'repair':[], 'retention':[]}
            for i,g in enumerate(leaves):
                for z in leaves[i+1:]:
                    if g['search_ply']!=z['search_ply'] or g['search_ply']<1:
                        continue
                    h = len(root['history'])
                    if g['history'][h] == z['history'][h]:
                        continue
                    good,bad = (g,z) if sign*(g['score_cp']-z['score_cp'])>0 else (z,g)
                    gap = sign*(good['score_cp']-bad['score_cp'])
                    margin = sign*(good['strategic_base_cp']-bad['strategic_base_cp'])
                    if gap<15 or abs(good['score_cp'])>1200 or abs(bad['score_cp'])>1200:
                        continue
                    maximum = attainable_margin_cp(*[torch.tensor(v) for v in
                        (good['strategic_base_cp'],bad['strategic_base_cp'],WEIGHT,WEIGHT,sign)],250).item()
                    if maximum<=0 or margin>150:
                        continue
                    kind = 'repair' if margin<=0 else 'retention'
                    options[kind].append({'good_fen':good['fen'],'bad_fen':bad['fen'],
                                          'sign':sign,'cp_loss':gap,'base_margin_cp':margin,
                                          'maximum_signed_margin_cp':maximum,'kind':kind,
                                          'game_id':root['game_id'],'family':root['family'],
                                          'root_index':ri,'root_fen':root['fen'],
                                          'search_ply':good['search_ply'],'label_kind':'static_leaf_ranking'})
            for kind in ('repair','retention'):
                rng.shuffle(options[kind]); accepted = 0
                for p in options[kind]:
                    endpoints = {key(p['good_fen']),key(p['bad_fen'])}
                    if endpoints & used:
                        continue
                    pairs.append(p); used.update(endpoints); accepted+=1
                    if accepted==2:
                        break
        output[split] = pairs
    assert len(output['train'])>=200 and sum(p['kind']=='repair' for p in output['train'])>=40, 'Too little ranking supervision; do not fit'
    write(path,output); return output


def retain_static(sf,data):
    path = ART/'ordinary-retention.json'
    if path.exists():
        return json.loads(path.read_text())
    used = {key(r['fen']) for rs in data['rows'].values() for r in rs}
    used.update(key(r['fen']) for rs in data['roots'].values() for r in rs)
    old = json.loads((ROOT/'ml/artifacts/ranking-v30/retained-score-rows.json').read_text())
    rng = random.Random(320036); rng.shuffle(old); rows = []; base = StrategicEvaluator(PRIORS)
    for old_row in old:
        b = chess.Board(old_row['fen']); alias = key(b.fen())
        if alias in used or not gate_active(b) or b.is_game_over():
            continue
        row = {'fen':b.fen(),'source':'legacy_position_relabelled_static','strategic_base_cp':base(b)}
        row.update(static_evaluation(sf,b)); rows.append(row); used.add(alias)
        if len(rows)==1000:
            break
    assert len(rows)==1000; write(path,rows); return rows


def tensor_scores(rows):
    return (torch.stack([torch.from_numpy(board_to_threat_array(chess.Board(r['fen']))) for r in rows]),
            torch.tensor([r['strategic_base_cp'] for r in rows],dtype=torch.float32),
            torch.tensor([r['score_cp'] for r in rows],dtype=torch.float32))


def tensor_pairs(pairs,rows):
    byfen = {r['fen']:r for r in rows}; arrays = [[] for _ in range(6)]
    for p in pairs:
        g,z = byfen[p['good_fen']],byfen[p['bad_fen']]
        vals = (torch.from_numpy(board_to_threat_array(chess.Board(g['fen']))),
                torch.from_numpy(board_to_threat_array(chess.Board(z['fen']))),
                g['strategic_base_cp'],z['strategic_base_cp'],p['sign'],p['cp_loss'])
        for a,v in zip(arrays,vals):
            a.append(v)
    return tuple(torch.stack(a) if i<2 else torch.tensor(a,dtype=torch.float32) for i,a in enumerate(arrays))


def metadata(state):
    return {'version':THREAT_VERSION,'input_size':THREAT_INPUT_SIZE,'score_scale':SCORE_SCALE,
            'baseline_id':'strategic-v26','baseline_weights':list(PRIORS),'training_correction_weight':WEIGHT,
            'training_quiet_only':False,'training_gate':GATE_ID,'training_target':'stockfish_static_white_cp',
            'correction_limit_cp':250,'hidden_sizes':[64,32],'state_dict':state}


def train_once(data,pairs,retention):
    torch.manual_seed(320037)
    scores = tensor_scores(data['rows']['train']+retention); ranks = tensor_pairs(pairs['train'],data['rows']['train'])
    all_importance = torch.tensor([3. if p['kind']=='repair' else 1. for p in pairs['train']])
    model = ThreatNet(); saved = torch.load(PREVIOUS,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['state_dict'])
    nn.init.zeros_(model.net[4].weight); nn.init.zeros_(model.net[4].bias)
    torch.save(metadata(copy.deepcopy(model.state_dict())),ART/'initial-zero-correction.pt')
    loader = DataLoader(TensorDataset(*scores),batch_size=128,shuffle=True)
    rankloader = DataLoader(TensorDataset(*ranks,all_importance),batch_size=128,shuffle=True)
    opt = torch.optim.AdamW(model.parameters(),lr=.0005); history = []; paths = []
    for epoch in range(1,25):
        model.train()
        for x,b,y in loader:
            opt.zero_grad(); delta = WEIGHT*model(x); prediction = rounded_score(b,delta,True)
            target_delta = ((y-b)/SCORE_SCALE).clamp(-250*WEIGHT/SCORE_SCALE,250*WEIGHT/SCORE_SCALE)
            loss = nn.functional.smooth_l1_loss(delta,target_delta,beta=.05)+4*nn.functional.mse_loss(torch.sigmoid(prediction/400),torch.sigmoid(y/400))+.02*delta.square().mean()
            loss.backward(); opt.step()
        for g,z,gb,zb,sign,cp,importance in rankloader:
            opt.zero_grad()
            gap = sign*(rounded_score(gb,WEIGHT*model(g),True)-rounded_score(zb,WEIGHT*model(z),True))
            ref = sign*(gb-zb)
            rankloss = nn.functional.binary_cross_entropy_with_logits(gap/100,torch.sigmoid(cp.clamp(max=500)/100),reduction='none')
            protect = torch.where(ref>0,torch.relu(ref.clamp(max=20)-gap)/100,torch.zeros_like(gap))
            ((rankloss*importance).mean()+10*protect.mean()).backward(); opt.step()
        if epoch in (8,16,24):
            path = ART/f'epoch-{epoch}.pt'; torch.save(metadata(copy.deepcopy(model.state_dict())),path); paths.append(path)
            model.eval()
            with torch.inference_mode():
                g,z,gb,zb,sign,cp = ranks
                gap = sign*(rounded_score(gb,WEIGHT*model(g))-rounded_score(zb,WEIGHT*model(z)))
                repairs = all_importance==3
                row = {'epoch':epoch,'correct':int((gap>0).sum()),'pairs':len(gap),
                       'repair_correct':int(((gap>0)&repairs).sum()),'repair_total':int(repairs.sum()),
                       'retention_regressions':int(((gap<=0)&~repairs).sum())}
                history.append(row); print('Training',row,flush=True)
    write(ART/'training.json',{'history':history,'one_trajectory':True,'epochs':24,
                             'score_labels':len(scores[0]),'static_pair_count':len(all_importance),
                             'initial_output_zero':True,'label_kind':'stockfish_static_white_cp'})
    return paths


def inference_audit(paths,data):
    candidates = [r for rs in data['rows'].values() for r in rs]
    positions = [r for category in ('capture','quiet') for r in [r for r in candidates if r['category']==category][:60]]
    results = {}
    for path in paths+[ART/'initial-zero-correction.pt']:
        model = StandPatHybrid(path); maximum = 0
        for r in positions:
            b = reconstruct(r); before = b.fen()
            with torch.inference_mode():
                expected = round(model.baseline(b)+WEIGHT*model.model(torch.from_numpy(board_to_threat_array(b))).item()*SCORE_SCALE)
            actual = model.evaluate_position(b); maximum=max(maximum,abs(actual-expected))
            assert abs(actual-expected)<=1 and actual==-model.evaluate_position(b.mirror()) and b.fen()==before
            if path.name=='initial-zero-correction.pt':
                assert actual==model.baseline(b)
        results[path.name] = {'positions':len(positions),'maximum_rounding_difference_cp':maximum,'color_symmetry':True}
    write(ART/'inference-audit.json',results)


def gate(summary,name):
    return common.qualifies(summary,name) and all(rs[name]['comparable']>=20 for rs in summary.values())


def pilot(path):
    import subprocess,sys
    opening = ROOT/'benchmarks/openings-standpat-v32-pilot.json'
    subprocess.run([sys.executable,'-m','benchmarks.generate_strength_openings','--output',str(opening),'--pairs','10','--seed','320038'],check=True)
    candidate = StandPatHybrid(path); base = StrategicEvaluator(PRIORS)
    def selector(name,b,time_limit,depth_cap):
        p = search(b,depth=64,time_limit=time_limit,eval_fn=candidate if name=='current' else base,use_lmr=True)
        stats=asdict(p); stats.pop('move'); return p.move,stats
    games=[]; pgns=''
    for i,o in enumerate(json.loads(opening.read_text()),1):
        for color in chess.COLORS:
            record,pgn=play_game(o,color,.25,600,64,selector,i,'new_heuristic'); games.append(record); pgns+=pgn
            summary=summarize(games)
            write(ART/'pilot/report.json',{'games':games,'summary':summary,'planned_games':20,'status':'running'})
            (ART/'pilot/games.pgn').write_text(pgns); print('Pilot',len(games),summary,flush=True)
            if summary['errors'] or summary['unfinished'] or summary['interrupted']:
                raise RuntimeError('Incomplete pilot; preserve results')
    write(ART/'pilot/report.json',{'games':games,'summary':summary,'planned_games':20,'status':'completed'})


def main():
    ART.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    protocol = {'gate':GATE_ID,'architecture':'Unchanged1420/64/32, raw250cp bound,25%blend; pretrained hidden layers, output resetzero before broader activation.',
                'labels':'Actual baseline static calls depth<=3/node2000;16unique quiet+16capture reservoir/search; teacher Final static eval Whitecp fromSF19 trace, not searched/settled values. No runtime Stockfish. Legacy1000positions relabelled static.',
                'sources':'960frozen bookstarts seed320032; exclude knownv29/v30/v31families/aliases; whole6plyfamilies split2:1:1; thirdsSFselfplay3000nodes, heuristic1500nodes,selfplay andvsSF;120sourceplies; roots16/48/80/112+coloroffset, teacher64k nonmate|Whitecp|<=800. Actual leafaliases unique acrossfolds.',
                'pairs':'Actual static leaves fromsame root, same searchply,different first rootmove;>=15cp staticteachergap, boundedfeasible, max2repair+2retention/root, endpoints neverreused; >=200trainpairs and>=40repairs before fitting. These arestaticrankings, notproved rootmove choices.',
                'training':'One24epoch trajectory, snapshots8/16/24; lr.0005; clippedstaticresidualSmoothL1 + roundedscoreproxy; repairimportance3,retentionprotection10,margincap20,correctionpenalty.02. No searched labels mixed into static targets.',
                'selection':'Freeze24fresh val/test roots each, one/game,max2/family,>=12families, teacheronlycoverage. Select searched depth3/250ms/depth64; nonworse depthmean,strictbettertimedmean,noextra>=150cperrors/losingtransitions/mates,common>=20. One selected checkpoint testonce. Pilot20 onlyaftertestpass. Appunchanged.',
                'limits':'Static teacher includes learned tactical knowledge, but doesnotexecute search. Broadergate increasesinferencecost. Legacyfamilyprovenanceincomplete. Targets and gate change together; noisolatedablation.',
                'previous_sha256':digest(PREVIOUS),'teacher_sha256':digest(SF),
                'source_hashes':{p:digest(ROOT/p) for p in ('engine/search.py','engine/strategic.py','ml/run_standpat_v32.py','ml/static_teacher.py','ml/standpat_hybrid.py','ml/threat_model.py','tools/stockfish-sf19/stockfish/src/evaluate.cpp','benchmarks/openings-standpat-v32-data.json')}}
    path=ART/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==protocol,'Frozen protocol changed'
    else:
        write(path,protocol)
    with exclusive_cpu('v32 actual static labels gate training and fresh screens'):
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            data=collect(sf); roots=freeze_screens(data); pairs=build_pairs(data); retention=retain_static(sf,data)
            paths=train_once(data,pairs,retention); inference_audit(paths,data)
            common.ART=ART
            models={'heuristic':StrategicEvaluator(PRIORS),'previous_quiet_nn':ThreatHybrid(PREVIOUS),
                    **{p.stem:StandPatHybrid(p) for p in paths}}
            development=common.screen(sf,roots['val'],models,'development')
            eligible=[p.stem for p in paths if gate(development,p.stem)]
            selected=min(eligible,key=lambda n:development['250ms'][n]['mean_regret']) if eligible else None
            write(ART/'selection.json',{'selected':selected,'test_used_for_selection':False})
            if not selected:
                write(ART/'decision.json',{'qualifies_for_games':False,'reason':'No snapshot passed fresh searched validation','app_unchanged':True});return
            test=common.screen(sf,roots['test'],{n:models[n] for n in ('heuristic',selected)},'test')
            passed=gate(test,selected)
            write(ART/'decision.json',{'selected':selected,'qualifies_for_games':passed,'app_unchanged':True})
        if passed:
            pilot(ART/f'{selected}.pt')


if __name__=='__main__':
    main()
