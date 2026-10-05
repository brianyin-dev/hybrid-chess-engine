"""Root-confirmed quiescence-leaf supervision and bounded output ablation."""
import copy,json,random,sys
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import chess,chess.engine
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import opening_board,play_game,summarize
from benchmarks.build_strategy_v26 import reconstruct
from engine.search import search,_SearchLimit,INF
from engine.strategic import StrategicEvaluator,PRIORS
from ml.decision_hybrid import DecisionNet,DecisionHybrid,metadata
from ml.standpat_hybrid import StandPatHybrid
from ml.threat_model import board_to_threat_array
from ml.model import SCORE_SCALE
from ml.train import rounded_score
from ml.generate_search_data import key
from ml.run_balanced_v19 import write,digest,SF
from ml.run_search_correction_v29 import family
from ml.run_search_v22 import exact
from ml.diagnose_critical_v23 import Trace
from ml.static_teacher import static_evaluation
from ml import run_strategic_v28 as common
from ml.run_standpat_v32 import known_provenance

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'ml/artifacts/decision-v33'
OLD=ROOT/'ml/artifacts/standpat-v32';INITIAL=OLD/'epoch-8.pt'
SPLITS=('train','val','test');TARGETS={'train':70,'val':30,'test':30}


def tags(b):
    out=[]
    if any(b.pieces_mask(chess.PAWN,c)&(chess.BB_RANK_6|chess.BB_RANK_7 if c else chess.BB_RANK_2|chess.BB_RANK_3) for c in chess.COLORS):out.append('advanced_pawn')
    if any(b.is_pinned(p.color,sq) for sq,p in b.piece_map().items() if p.piece_type!=chess.KING):out.append('pin')
    if b.is_check() or any(b.attackers(not c,b.king(c)) or any(b.attackers(not c,s) for s in chess.scan_forward(chess.BB_KING_ATTACKS[b.king(c)])) for c in chess.COLORS):out.append('king_pressure')
    if any(m.promotion for m in b.legal_moves):out.append('immediate_promotion')
    return out


def freeze():
    if (ART/'allocation.json').exists():return json.loads((ART/'allocation.json').read_text())
    previous=json.loads((OLD/'allocation.json').read_text());owners={o['family']:s for s,os in previous.items() for o in os}
    _,oldfamilies=known_provenance();assigned={s:[] for s in SPLITS};counts=Counter()
    openings=json.loads((ROOT/'benchmarks/openings-decision-v33-data.json').read_text());random.Random(330034).shuffle(openings)
    groups=defaultdict(list)
    for o in openings:
        f=family(o['moves'])
        if f not in oldfamilies:groups[f].append(o)
    for f,os in sorted(groups.items(),key=lambda item:(item[0] not in owners,item[0])):
        s=owners.get(f) or min(SPLITS,key=lambda z:len(assigned[z])/TARGETS[z]);owners[f]=s
        for o in os[:3]:
            if len(assigned[s])<TARGETS[s]:assigned[s].append({**o,'family':f})
    assert all(len(assigned[s])>=TARGETS[s] for s in SPLITS),{s:len(os) for s,os in assigned.items()}
    write(ART/'allocation.json',assigned);return assigned


def forbidden():
    aliases,_=known_provenance()
    d=json.loads((OLD/'collection.json').read_text())
    for rs in d['rows'].values():aliases.update(key(r['fen']) for r in rs)
    for rs in d['roots'].values():aliases.update(key(r['fen']) for r in rs)
    match=json.loads((OLD/'user-pilot-20/report.json').read_text())
    for g in match['games']:
        b=chess.Board(g['initial_fen'])
        for m in g['opening_moves']:b.push_uci(m)
        aliases.add(key(b.fen()))
        for m in g['moves']:b.push_uci(m['uci']);aliases.add(key(b.fen()))
    return aliases


def collect_sources(sf):
    path=ART/'sources.json';data=json.loads(path.read_text()) if path.exists() else {'sources':[],'roots':{s:[] for s in SPLITS},'complete':False}
    if data['complete']:return data
    seen=forbidden()|{key(r['fen']) for rs in data['roots'].values() for r in rs};base=StrategicEvaluator(PRIORS)
    for s,os in freeze().items():
        for gi,o in enumerate(os):
            gid=3300000+10000*SPLITS.index(s)+gi
            if any(g['game_id']==gid for g in data['sources']):continue
            b=opening_board(o);ordinary=[];targeted=[];last_target=-99
            for ply in range(144):
                if b.is_game_over():break
                ts=tags(b)
                candidate=ply in (16+gi%2,40+gi%2,64+gi%2,88+gi%2,112+gi%2,136+gi%2)
                targeted_slot=bool(set(ts)&{'advanced_pawn','pin'}) and ply>=20 and ply-last_target>=16 and len(targeted)<3
                alias=key(b.fen())
                if (candidate or targeted_slot) and alias not in seen:
                    r={'fen':b.fen(),'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack],
                       'game_id':gid,'family':o['family'],'opening':o,'split':s,'source_ply':ply,'tags':ts,
                       'screen_stage':'endgame' if len(b.piece_map())<=12 else 'middlegame'}
                    (ordinary if candidate else targeted).append(r);seen.add(alias)
                    if targeted_slot:last_target=ply
                ours=gi%3==1 and b.turn==(gi%2==0)
                move=search(b,depth=64,node_limit=750,eval_fn=base,use_lmr=True).move if ours else exact(sf,b,nodes=3000)['pv'][0]
                b.push(move)
            data['sources'].append({'game_id':gid,'family':o['family'],'split':s,'opening':o,'history':[m.uci() for m in b.move_stack],
                                    'final_fen':b.fen(),'truncated':not b.is_game_over(),'policy':'heuristic_vs_stockfish' if gi%3==1 else 'stockfish_selfplay'})
            data['roots'][s]+=ordinary+targeted;write(path,data)
            print('Source',s,gi+1,len(os),'roots',len(data['roots'][s]),flush=True)
    data['complete']=True;write(path,data);return data


def leaf(root,move,base):
    b=reconstruct(root);worker=Trace(b,base,None,False,node_limit=8000)
    # Full-window diagnostic branch, fixed depth2 followed by tactical quiescence.
    try:
        with worker.pushed(b,move):score,r=worker.ntrace(b,1,-INF,INF,1)
    except _SearchLimit:return None
    if r is None:return None
    position=chess.Board(r['fen'])
    if position.is_check() or position.is_game_over():return None
    white=-score*(1 if b.turn else -1)
    if white!=base.evaluate_position(position):return None
    return {'fen':r['fen'],'initial_fen':root['initial_fen'],'history':r['history'],
            'strategic_base_cp':white,'game_id':root['game_id'],'family':root['family'],'split':root['split'],
            'root_fen':root['fen'],'root_move':move.uci(),'tags':tags(position),'trace_nodes':worker.nodes,'trace_qnodes':worker.qnodes}


def build_data(sf,sources):
    path=ART/'collection.json';d=json.loads(path.read_text()) if path.exists() else {
        'rows':{s:[] for s in SPLITS},'pairs':{s:[] for s in SPLITS},'screen_roots':{s:[] for s in SPLITS},'processed':[], 'rejects':{},'complete':False}
    if d['complete']:return d
    base=StrategicEvaluator(PRIORS);used=forbidden()|{key(r['fen']) for rs in d['rows'].values() for r in rs}
    rejects=Counter(d['rejects']);owner={key(r['fen']):s for s,rs in d['rows'].items() for r in rs}
    for s,roots in sources['roots'].items():
        # Interleave games so a stop at fixed quotas cannot select many related roots.
        ordered=sorted(roots,key=lambda r:(r['source_ply'],r['game_id']))
        for i,r in enumerate(ordered):
            rid=f"{s}:{r['game_id']}:{r['source_ply']}"
            if rid in d['processed']:continue
            b=reconstruct(r);sign=1 if b.turn else -1
            teacher=exact(sf,b,nodes=64000);rootcp=teacher['score'].white().score()
            if rootcp is not None and abs(rootcp)<=800:d['screen_roots'][s].append({**r,'teacher_root_white_cp':rootcp})
            if rootcp is None or abs(rootcp)>1200:
                rejects['root_mate_or_extreme']+=1;d['processed'].append(rid);continue
            goodmove=teacher['pv'][0]
            h=search(b,depth=2,node_limit=2500,eval_fn=base,use_lmr=True)
            infos=sf.analyse(b,chess.engine.Limit(nodes=32000),multipv=min(3,b.legal_moves.count()),game=object())
            alternatives=[]
            for m in [h.move]+[a['pv'][0] for a in infos if a.get('pv')]:
                if m and m!=goodmove and m not in alternatives:alternatives.append(m)
            g=leaf(r,goodmove,base)
            if g is None:
                rejects['good_trace_unsettled_or_limit']+=1;d['processed'].append(rid);continue
            glabel=static_evaluation(sf,reconstruct(g));g.update(glabel)
            quickgood=exact(sf,b,[goodmove],nodes=128000)['score'].pov(b.turn).score()
            accepted=0
            for move in alternatives[:3]:
                z=leaf(r,move,base)
                if z is None:rejects['bad_trace_unsettled_or_limit']+=1;continue
                if key(g['fen'])==key(z['fen']) or key(g['fen']) in used or key(z['fen']) in used:
                    rejects['duplicate_or_prior_endpoint']+=1;continue
                z.update(static_evaluation(sf,reconstruct(z)))
                staticgap=sign*(g['score_cp']-z['score_cp']);margin=sign*(g['strategic_base_cp']-z['strategic_base_cp'])
                if staticgap<15 or abs(g['score_cp'])>1200 or abs(z['score_cp'])>1200:
                    rejects['static_disagrees_or_extreme']+=1;continue
                if margin<=-125 or margin>100:
                    rejects['infeasible_or_already_easy']+=1;continue
                badroot=exact(sf,b,[move],nodes=128000)['score'].pov(b.turn).score()
                if quickgood is None or badroot is None or quickgood-badroot<25:
                    rejects['no_consequential_root_gap']+=1;continue
                endpoints=[]
                settled=True
                for endpoint in (g,z):
                    eb=reconstruct(endpoint);a=exact(sf,eb,nodes=64000)
                    value=a['score'].white().score();best=a['pv'][0]
                    # Search settles exchanges; reject leaves whose best reply is still tactical.
                    if value is None or eb.is_capture(best) or best.promotion or eb.gives_check(best):settled=False
                    endpoints.append({'white_cp':value,'best_move':best.uci(),'depth':a.get('depth')})
                if not settled or sign*(endpoints[0]['white_cp']-endpoints[1]['white_cp'])<15:
                    rejects['endpoint_search_unsettled_or_reverses']+=1;continue
                confirmed=[]
                for m in (goodmove,move):
                    a=exact(sf,b,[m],nodes=256000);confirmed.append({'mover_cp':a['score'].pov(b.turn).score(),'depth':a.get('depth'),'move':m.uci()})
                if any(a['mover_cp'] is None for a in confirmed) or confirmed[0]['mover_cp']-confirmed[1]['mover_cp']<25:
                    rejects['root_confirmation_reverses']+=1;continue
                kind='repair' if margin<=0 else 'retention'
                p={'good_fen':g['fen'],'bad_fen':z['fen'],'root_fen':r['fen'],'root_history':r['history'],
                   'sign':sign,'cp_loss':confirmed[0]['mover_cp']-confirmed[1]['mover_cp'],'static_gap_cp':staticgap,
                   'base_margin_cp':margin,'kind':kind,'game_id':r['game_id'],'family':r['family'],
                   'tags':sorted(set(r['tags']+g['tags']+z['tags'])),'root_confirmation':confirmed,'endpoint_search':endpoints,
                   'label_kind':'root_confirmed_qtrace_leaf_ranking'}
                d['pairs'][s].append(p);d['rows'][s]+=[g,z];used.update((key(g['fen']),key(z['fen'])));accepted+=1
                # One pair per root, unique endpoints; roots throughout independent source games.
                break
            d['processed'].append(rid);d['rejects']=dict(rejects)
            if len(d['processed'])%10==0:
                write(path,d);print('Rank',s,i+1,'/',len(ordered),'pairs',len(d['pairs'][s]),'repairs',sum(p['kind']=='repair' for p in d['pairs'][s]),flush=True)
    d['complete']=True;d['rejects']=dict(rejects);write(path,d);return d


def screens(d):
    path=ART/'fresh-roots.json'
    if path.exists():return json.loads(path.read_text())
    result={}
    for s in ('val','test'):
        pool=copy.deepcopy(d['screen_roots'][s]);random.Random(330035+(s=='test')).shuffle(pool)
        chosen=[];games=set();families=Counter();colors=Counter();stages=Counter()
        while len(chosen)<24:
            options=[r for r in pool if r['game_id'] not in games and families[r['family']]<2]
            if not options:raise ValueError('Insufficient independent screen coverage')
            r=min(options,key=lambda x:(families[x['family']],colors[chess.Board(x['fen']).turn],stages[x['screen_stage']]))
            chosen.append(r);games.add(r['game_id']);families[r['family']]+=1;colors[chess.Board(r['fen']).turn]+=1;stages[r['screen_stage']]+=1
        result[s]=chosen
    assert not {r['family'] for r in result['val']}&{r['family'] for r in result['test']}
    write(path,result);return result


def tensors(rows,pairs):
    features={r['fen']:torch.from_numpy(board_to_threat_array(chess.Board(r['fen']))) for r in rows}
    byfen={r['fen']:r for r in rows}
    scores=(torch.stack([features[r['fen']] for r in rows]),torch.tensor([r['strategic_base_cp'] for r in rows],dtype=torch.float32),torch.tensor([r['score_cp'] for r in rows],dtype=torch.float32))
    ranks=(torch.stack([features[p['good_fen']] for p in pairs]),torch.stack([features[p['bad_fen']] for p in pairs]),
           *[torch.tensor([byfen[p[f]]['strategic_base_cp'] for p in pairs],dtype=torch.float32) for f in ('good_fen','bad_fen')],
           torch.tensor([p['sign'] for p in pairs],dtype=torch.float32),torch.tensor([3 if p['kind']=='repair' else 1 for p in pairs],dtype=torch.float32))
    return scores,ranks


def static_metrics(model,rows,pairs):
    model.eval();pred={};sat=0
    with torch.inference_mode():
        for r in rows:
            x=torch.from_numpy(board_to_threat_array(chess.Board(r['fen'])))
            delta=.25*model(x).item()*SCORE_SCALE;pred[r['fen']]=round(r['strategic_base_cp']+delta);sat+=abs(delta)>=60
    correct=repairs=damage=0
    for p in pairs:
        gap=p['sign']*(pred[p['good_fen']]-pred[p['bad_fen']]);correct+=gap>0
        repairs+=gap>0 and p['kind']=='repair';damage+=gap<=0 and p['kind']=='retention'
    return {'correct':correct,'pairs':len(pairs),'repair_correct':repairs,'retention_damage':damage,'near_bound_endpoints':sat,'endpoints':len(rows)}


def fit(d):
    old=json.loads((OLD/'collection.json').read_text())['rows']['train']
    held={key(r['fen']) for s in ('val','test') for r in d['rows'][s]+d['screen_roots'][s]}
    retention=[r for r in old if key(r['fen']) not in held]
    random.Random(330036).shuffle(retention);retention=retention[:2000]
    allrows=d['rows']['train']+retention
    score,ranks=tensors(allrows,d['pairs']['train'])
    paths=[];history={};initial=torch.load(INITIAL,map_location='cpu',weights_only=True)['state_dict']
    for mode in ('tanh','linear_clip'):
        torch.manual_seed(330037);model=DecisionNet(mode);model.load_state_dict(initial)
        nn.init.zeros_(model.net[4].weight);nn.init.zeros_(model.net[4].bias)
        torch.save(metadata(copy.deepcopy(model.state_dict()),mode),ART/f'{mode}-zero.pt')
        opt=torch.optim.AdamW(model.parameters(),lr=.0005)
        loader=DataLoader(TensorDataset(*score),batch_size=128,shuffle=True)
        rankloader=DataLoader(TensorDataset(*ranks),batch_size=64,shuffle=True);records=[]
        for epoch in range(1,25):
            model.train()
            for x,b,y in loader:
                opt.zero_grad();delta=.25*model(x)
                # Supporting static labels capped inside the deployed bound, not driven to saturation.
                target=((y-b)/SCORE_SCALE).clamp(-50/SCORE_SCALE,50/SCORE_SCALE)
                raw=.25*model.raw(x)
                loss=.25*nn.functional.smooth_l1_loss(delta,target,beta=.05)+.1*torch.relu(raw.abs()-50/SCORE_SCALE).square().mean()
                loss.backward();opt.step()
            for g,z,gb,zb,sign,importance in rankloader:
                opt.zero_grad();gd=.25*model(g);zd=.25*model(z)
                gap=sign*(rounded_score(gb,gd,True)-rounded_score(zb,zd,True));ref=sign*(gb-zb)
                target=torch.where(ref>0,ref.clamp(max=20),torch.full_like(ref,20))
                violation=torch.relu(target-gap)/100
                protection=torch.where(ref>0,torch.relu(ref.clamp(max=20)-gap)/100,torch.zeros_like(gap))
                rawg=.25*model.raw(g);rawz=.25*model.raw(z)
                loss=(violation*importance).mean()+10*protection.mean()+.1*(torch.relu(rawg.abs()-50/SCORE_SCALE).square().mean()+torch.relu(rawz.abs()-50/SCORE_SCALE).square().mean())
                loss.backward();opt.step()
            if epoch in (8,16,24):
                path=ART/f'{mode}-epoch-{epoch}.pt';torch.save(metadata(copy.deepcopy(model.state_dict()),mode),path);paths.append(path)
                row={'epoch':epoch,'train':static_metrics(model,d['rows']['train'],d['pairs']['train']),
                     'validation':static_metrics(model,d['rows']['val'],d['pairs']['val'])};records.append(row)
                print('Train',mode,row,flush=True)
        history[mode]=records
    write(ART/'training.json',{'arms':history,'epochs':24,'seed':330037,'legacy_static_retention':len(retention),
                             'objective':'20cp feasible hinge rankings + 10x correct-decision protection; supporting static residual target capped at50cp; raw-output excess penalty; exact noncheck25% hybrid rounding.'})
    # Predeclared static development screen chooses one checkpoint per arm for searched validation.
    chosen=[]
    for mode,records in history.items():
        row=min(records,key=lambda r:(r['validation']['retention_damage'],-r['validation']['correct'],r['validation']['near_bound_endpoints'],r['epoch']))
        chosen.append(ART/f"{mode}-epoch-{row['epoch']}.pt")
    write(ART/'static-selection.json',{'chosen':[p.name for p in chosen],'policy':'Retention damage first, total correct second, saturation third, earliest epoch tie; development only, not test.'})
    return paths,chosen


def audit_runtime(paths,d):
    rows=d['rows']['train'][:60]+d['rows']['val'][:60]
    out={}
    for path in paths+[ART/'tanh-zero.pt',ART/'linear_clip-zero.pt']:
        adapter=DecisionHybrid(path);maximum=0
        for r in rows:
            b=reconstruct(r);before=b.fen();base=adapter.baseline(b)
            with torch.inference_mode():expected=round(base+.25*adapter.model(torch.from_numpy(board_to_threat_array(b))).item()*SCORE_SCALE)
            actual=adapter.evaluate_position(b);maximum=max(maximum,abs(expected-actual))
            assert maximum<=1 and actual==-adapter.evaluate_position(b.mirror()) and b.fen()==before
            assert abs(actual-base)<=63
            if path.stem.endswith('zero'):assert actual==base
        out[path.name]={'positions':len(rows),'maximum_torch_runtime_difference_cp':maximum,'bound_color_board_checks':True}
    write(ART/'inference-audit.json',out)


def qualified(summary,name):
    return common.qualifies(summary,name) and all(rs[name]['comparable']>=20 for rs in summary.values())


def main():
    ART.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);common.ART=ART
    if (ART/'decision.json').exists():raise FileExistsError('Completed experiment; preserve results')
    protocol={'architecture':'Same1420/64/32; same features, noncheck gate, weight25%, rawbound250cp; no speed/search changes.',
              'output_ablation':['tanh','linear_clip'],'training_epochs':24,'snapshots':[8,16,24],
              'supervision':'Full-window HCE forced depth2+quiescence score-bearing leaves. Root move preference confirmed128k then256k SF; leaves agree static and64k searched, best SF reply quiet noncheck/nonpromotion. Unique endpoints, one pair/root.',
              'sources':TARGETS,'minimum_train_pairs':120,'minimum_train_repairs':20,'minimum_validation_test_pairs':30,
              'selection':'One checkpoint/arm by development retention damage/correct/saturation; compare searched depth3 and250ms on24 fresh val positions. Require >=20comparable, nonworse mean depth3, strictly better timed, no extra150cp errors/losing transitions/mates. If passes, test chosen once on24 untouched positions with same gate. Games20 only if passes.',
              'split':'Whole6ply canonical families retain v32 split assignments; new families allocated by target ratios. Fresh trajectories/roots exclude old aliases; no claims these are wholly unseen historical opening families. Warm model/retention provenance inherited.',
              'initialization':'v32ep8 hidden layers, final head resetzero identically in both arms; same seeds/data/order.',
              'source_sha256':{p:digest(ROOT/p) for p in ['engine/search.py','engine/strategic.py','ml/decision_hybrid.py','ml/run_decision_v33.py']},
              'initial_sha256':digest(INITIAL),'app_unchanged':True}
    if not (ART/'protocol.json').exists():write(ART/'protocol.json',protocol)
    with exclusive_cpu('v33 settled supervision output-shaping experiment'):
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
            sources=collect_sources(sf);d=build_data(sf,sources);fresh=screens(d)
            counts={s:{'pairs':len(d['pairs'][s]),'repairs':sum(p['kind']=='repair' for p in d['pairs'][s]),
                       'games':len({p['game_id'] for p in d['pairs'][s]}),'families':len({p['family'] for p in d['pairs'][s]}),
                       'tags':dict(Counter(t for p in d['pairs'][s] for t in p['tags']))} for s in SPLITS}
            write(ART/'data-summary.json',counts);print('Data',counts,flush=True)
            if counts['train']['pairs']<120 or counts['train']['repairs']<20 or min(counts[s]['pairs'] for s in ('val','test'))<30:
                write(ART/'decision.json',{'trained':False,'reason':'Predeclared dataset coverage gate failed','counts':counts,'qualifies_for_games':False,'app_unchanged':True});return
            paths,chosen=fit(d);audit_runtime(paths,d)
            models={'heuristic':StrategicEvaluator(PRIORS),'old_v32':StandPatHybrid(INITIAL),**{p.stem:DecisionHybrid(p) for p in chosen}}
            development=common.screen(sf,fresh['val'],models,'development')
            eligible=[p for p in chosen if qualified(development,p.stem)]
            selected=min(eligible,key=lambda p:development['250ms'][p.stem]['mean_regret']) if eligible else None
            write(ART/'selection.json',{'selected':selected.name if selected else None,'eligible':[p.name for p in eligible]})
            if selected:
                test=common.screen(sf,fresh['test'],{'heuristic':StrategicEvaluator(PRIORS),'old_v32':StandPatHybrid(INITIAL),selected.stem:DecisionHybrid(selected)},'test')
                passed=qualified(test,selected.stem)
            else:test=None;passed=False
            write(ART/'decision.json',{'trained':True,'selected':selected.name if selected else None,'qualifies_for_games':passed,'app_unchanged':True})
        if passed:
            # A short paired match only; never automatically expand.
            import subprocess
            opening=ROOT/'benchmarks/openings-decision-v33-pilot.json'
            subprocess.run([sys.executable,'-m','benchmarks.generate_strength_openings','--output',str(opening),'--pairs','10','--seed','330039'],check=True)
            games=[];pgns='';candidate=DecisionHybrid(selected);base=StrategicEvaluator(PRIORS)
            def selector(name,b,t,depth):
                start=perf_counter();r=search(b,depth=depth,time_limit=t,eval_fn=candidate if name=='current' else base,use_lmr=True)
                stats=asdict(r);stats.pop('move');stats['elapsed']=perf_counter()-start;stats['budget_seconds']=t;stats['overrun_seconds']=max(0,stats['elapsed']-t)
                return r.move,stats
            for i,o in enumerate(json.loads(opening.read_text()),1):
                for color in (chess.WHITE,chess.BLACK):
                    g,pgn=play_game(o,color,.25,600,64,selector,i,'new_heuristic');games.append(g);pgns+=pgn
                    write(ART/'pilot/report.json',{'games':games,'summary':summarize(games),'planned_games':20,'status':'running' if len(games)<20 else 'completed'})
                    (ART/'pilot/games.pgn').write_text(pgns);print('Pilot',len(games),summarize(games),flush=True)
                    if g['error'] or g['reason']=='interrupted':raise RuntimeError('Pilot error; preserve results')

if __name__=='__main__':main()
