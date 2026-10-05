"""Search-distribution supervision for the unchanged strategic hybrid."""
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from benchmarks.match import opening_board
from engine.search import search
from engine.strategic import PRIORS, StrategicEvaluator
from ml import run_strategic_v28 as previous
from ml.dataset import correction_factor
from ml.diagnose_critical_v23 import review, forced
from ml.confirm_fatal_v23 import fatal
from ml.generate_search_data import key
from ml.model import RELATIONAL_INPUT_SIZE, SCORE_SCALE, board_to_tensor
from ml.strategic_hybrid import StrategicHybrid
from ml.train import rounded_score, attainable_margin_cp
from ml.run_balanced_v19 import write, SF, digest
from ml.run_search_v22 import exact

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/search-correction-v29'
OLD=ROOT/'ml/artifacts/strategic-v28'


def family(history,initial=chess.STARTING_FEN):
    b=chess.Board(initial)
    for move in history[:6]:b.push_uci(move)
    return key(b.fen())


def quiet(b):
    return not b.is_game_over() and bool(correction_factor(b,1,True))


class Leaves:
    cacheable_by_fen=True
    position_only=True
    def __init__(self,seed):
        self.baseline=StrategicEvaluator(PRIORS)
        self.rng=random.Random(seed);self.rows=[];self.count=0
    def evaluate_position(self,b):
        if quiet(b):
            self.count+=1
            row={'fen':b.fen(),'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack]}
            if len(self.rows)<24:self.rows.append(row)
            else:
                i=self.rng.randrange(self.count)
                if i<24:self.rows[i]=row
        return self.baseline(b)
    __call__=evaluate_position


def settle(sf,root,move):
    b=reconstruct(root);b.push(move)
    for _ in range(12):
        if b.is_game_over():return None
        if quiet(b):break
        b.push(exact(sf,b,nodes=8000)['pv'][0])
    if not quiet(b):return None
    info=exact(sf,b,nodes=64000);cp=info['score'].white().score()
    if cp is None:return None
    return {'fen':b.fen(),'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack],
            'score_cp':cp,'teacher_nodes':64000,'settling_nodes_per_step':8000}


def interval(g,z,sign,weight):
    base=StrategicEvaluator(PRIORS)
    gb,zb=base(chess.Board(g['fen'])),base(chess.Board(z['fen']))
    gf,zf=weight*correction_factor(chess.Board(g['fen']),1,True),weight*correction_factor(chess.Board(z['fen']),1,True)
    tensors=[torch.tensor(v) for v in (gb,zb,gf,zf,sign)]
    upper=attainable_margin_cp(*tensors,250).item()
    return {'base_margin_cp':sign*(gb-zb),'maximum_signed_margin_cp':upper,'permits_correct_ranking':upper>0,
            'limit_per_endpoint_cp':250*weight,'scope':'Endpoint bound; permits but does not prove a searched root move repair.'}


def diagnose(sf):
    records=json.loads((OLD/'development.json').read_text());outputs=[];critical_rows=[];critical_pairs=[]
    for i,r in enumerate(records):
        root=r['root'];sign=1 if reconstruct(root).turn else -1
        result={'root':root,'cases':[]}
        for mode,probes in r['probes'].items():
            base=probes['heuristic'];br=base['review']
            for name in ('0.1','0.25'):
                p=probes[name];pr=p['review']
                if p['move']==base['move']:continue
                bad=pr['cp_loss'];reference=br['cp_loss']
                consequential=(bad is not None and reference is not None and bad-reference>=50) or (fatal(pr) and not fatal(br)) or (pr['allows_mate'] and not br['allows_mate'])
                case={'mode':mode,'weight':float(name),'heuristic_move':base['move'],'nn_move':p['move'],
                      'heuristic_review':br,'nn_review':pr,'consequential_regression':bool(consequential)}
                if consequential:
                    # Full-window reference traces diagnose leaf/horizon differences.
                    traces={n:forced(reconstruct(root),chess.Move.from_uci(m),StrategicEvaluator(PRIORS),3)
                            for n,m in [('heuristic',base['move']),('nn',p['move'])]}
                    case['full_window_depth3_traces']=traces
                    good_move=chess.Move.from_uci(pr['best_move'])
                    good=settle(sf,root,good_move);bad_leaf=settle(sf,root,chess.Move.from_uci(p['move']))
                    if good and bad_leaf:
                        gap=sign*(good['score_cp']-bad_leaf['score_cp'])
                        case['settled_teacher_gap_cp']=gap
                        case['bounded_endpoints']={str(w):interval(good,bad_leaf,sign,w) for w in (.1,.25)}
                        critical_rows += [{**leaf,'game_id':2900000+i,'source':'confirmed_v28_regression'} for leaf in (good,bad_leaf)]
                        if gap>0:
                            critical_pairs.append({'good_fen':good['fen'],'bad_fen':bad_leaf['fen'],'root_fen':root['fen'],
                                'sign':sign,'cp_loss':gap,'game_id':2900000+i,'source':'settled_regression',
                                'feasible_weights':[w for w in (.1,.25) if case['bounded_endpoints'][str(w)]['permits_correct_ranking']]})
                result['cases'].append(case)
        outputs.append(result);write(ART/'diagnostics.json',outputs)
        print('Diagnosed',i+1,'/',len(records),flush=True)
    write(ART/'diagnostic-summary.json',{'roots':len(outputs),'disagreements':sum(len(r['cases']) for r in outputs),
        'consequential_regressions':sum(c['consequential_regression'] for r in outputs for c in r['cases']),
        'settled_pairs':len(critical_pairs),'feasible_pairs':{str(w):sum(w in p['feasible_weights'] for p in critical_pairs) for w in (.1,.25)},
        'diagnosed_roots_are_development':True,'full_window_traces':'Different diagnostic tree than LMR search; not a causal proof.'})
    return critical_rows,critical_pairs


def collect(sf,forbidden,critical_roots):
    openings=json.loads((ROOT/'benchmarks/openings-search-correction-v29.json').read_text())
    groups=defaultdict(list)
    for o in openings:groups[family(o['moves'])].append(o)
    excluded={family(r['history'],r['initial_fen']) for r in critical_roots}
    # Reserve the untouched v28 searched-test families too.
    reserved=json.loads((OLD/'searched-roots.json').read_text())['test']
    excluded.update(family(r['history'],r['initial_fen']) for r in reserved)
    assigned={'train':[],'val':[],'test':[]};targets={'train':24,'val':8,'test':8}
    for family_id,items in sorted(groups.items(),key=lambda x:(-len(x[1]),x[0])):
        if family_id in excluded:continue
        split=min(assigned,key=lambda s:len(assigned[s])/targets[s]);assigned[split]+=items
    assert all(len(assigned[s])>=targets[s] for s in targets),'Insufficient fresh family coverage'
    result={s:[] for s in targets};labels={s:[] for s in targets};pairs={s:[] for s in targets};seen=set(forbidden)
    baseline=StrategicEvaluator(PRIORS);source=[]
    for split in targets:
        for gi,o in enumerate(assigned[split][:targets[split]]):
            gid=2910000+list(targets).index(split)*1000+gi;b=opening_board(o);roots=[]
            # Self-play the actual new heuristic, not the teacher, at fixed nodes.
            for ply in range(80):
                if b.is_game_over():break
                if ply in (12,32,52,72):
                    row={'fen':b.fen(),'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack],
                         'game_id':gid,'family':family(o['moves']),'split':split}
                    if key(b.fen()) not in seen:roots.append(row);seen.add(key(b.fen()))
                p=search(b,depth=64,node_limit=1500,eval_fn=baseline,use_lmr=True)
                if p.move is None:break
                b.push(p.move)
            source.append({'game_id':gid,'split':split,'family':family(o['moves']),'opening':o,'history':[m.uci() for m in b.move_stack],'final_fen':b.fen()})
            # Two roots/game held out; all four can contribute training leaves.
            chosen=roots if split=='train' else roots[:2]
            result[split]+=chosen
            if split=='train':
                for ri,root in enumerate(chosen):
                    collector=Leaves(gid+ri);search(reconstruct(root),depth=3,node_limit=4000,eval_fn=collector,use_lmr=True)
                    collector.rng.shuffle(collector.rows)
                    for leaf in collector.rows[:4]:
                        alias=key(leaf['fen'])
                        if alias in seen:continue
                        cp=exact(sf,reconstruct(leaf),nodes=64000)['score'].white().score()
                        if cp is None:continue
                        seen.add(alias);labels[split].append({**leaf,'score_cp':cp,'game_id':gid,'family':root['family'],'source':'new_heuristic_quiet_search','teacher_nodes':64000})
                # Consequential disagreements from one representative root/game.
                if chosen:
                    root=chosen[-1];rb=reconstruct(root)
                    h=search(rb,depth=3,eval_fn=baseline,use_lmr=True)
                    old=StrategicHybrid(OLD/'weight-25.pt',.25);n=search(rb,depth=3,eval_fn=old,use_lmr=True)
                    if n.move!=h.move:
                        hr,nr=review(sf,rb,h.move,256000),review(sf,rb,n.move,256000)
                        if (nr['cp_loss'] or 0)>(hr['cp_loss'] or 0)+50:
                            g,z=settle(sf,root,chess.Move.from_uci(nr['best_move'])),settle(sf,root,n.move)
                            if g and z and (1 if rb.turn else -1)*(g['score_cp']-z['score_cp'])>0:
                                sign=1 if rb.turn else -1
                                pairs[split].append({'good_fen':g['fen'],'bad_fen':z['fen'],'root_fen':root['fen'],'game_id':gid,'sign':sign,
                                    'cp_loss':sign*(g['score_cp']-z['score_cp']),'feasible_weights':[w for w in (.1,.25) if interval(g,z,sign,w)['permits_correct_ranking']]})
                                labels[split]+=[{**r,'game_id':gid,'family':root['family'],'source':'new_disagreement'} for r in (g,z)]
            print('New-engine source',split,gi+1,'/',targets[split],'labels',len(labels[split]),flush=True)
            write(ART/'source-games.json',source);write(ART/'new-labels.json',labels);write(ART/'new-pairs.json',pairs);write(ART/'roots.json',result)
    assert len(result['val'])>=12 and len(result['test'])>=12
    familysets={s:{r['family'] for r in rs} for s,rs in result.items()}
    assert all(not familysets[a]&familysets[b] for a,b in [('train','val'),('train','test'),('val','test')])
    # Filter train labels/endpoints against every held-out root alias.
    heldout={key(r['fen']) for s in ('val','test') for r in result[s]}
    labels['train']=[r for r in labels['train'] if key(r['fen']) not in heldout]
    pairs['train']=[r for r in pairs['train'] if not any(key(r[f]) in heldout for f in ('good_fen','bad_fen','root_fen'))]
    write(ART/'new-labels.json',labels);write(ART/'new-pairs.json',pairs)
    return result,labels,pairs


def training_data(rows,pairs,roots,newlabels,newpairs,criticalrows,criticalpairs):
    rng=random.Random(290029);base=StrategicEvaluator(PRIORS)
    heldout={key(r['fen']) for s in ('val','test') for r in roots[s]}
    # Retain broad old labels and old correct rankings, rather than train only failures.
    oldrows=[r for r in rows['train'] if key(r['fen']) not in heldout];rng.shuffle(oldrows)
    oldpairs=[r for r in pairs['train'] if not any(key(r[f]) in heldout for f in ('good_fen','bad_fen','root_fen') if f in r)];rng.shuffle(oldpairs)
    fresh=[r for r in newlabels['train']+criticalrows if key(r['fen']) not in heldout and quiet(chess.Board(r['fen']))]
    for r in fresh:
        r['strategic_base_cp']=base(chess.Board(r['fen']));r['strategic_residual_cp']=r['score_cp']-r['strategic_base_cp']
    mix=copy.deepcopy(rows);mix['train']=oldrows[:2000]+fresh*4
    pairmix=copy.deepcopy(pairs);pairmix['train']=oldpairs[:1500]
    write(ART/'training-mix.json',{'ordinary_retention_rows':min(2000,len(oldrows)),'new_unique_score_rows':len({key(r['fen']) for r in fresh}),
        'new_row_sampling_multiplier':4,'retention_pairs':len(pairmix['train']),'new_pairs':len(newpairs['train']+criticalpairs),
        'labels':fresh,'ranking_examples':newpairs['train']+criticalpairs,'heldout_root_aliases':len(heldout),
        'split_limits':'Fresh source games/families separated; old family provenance incomplete. Diagnosed19 roots are training/development, not test.'})
    return mix,pairmix,newpairs['train']+criticalpairs


def train_candidates(weight,rows,pairs,focused):
    model=StrategicHybrid(OLD/f'weight-{int(weight*100)}.pt',weight).neural.model
    localpairs=copy.deepcopy(pairs)
    feasible=[p for p in focused if weight in p.get('feasible_weights',[])];localpairs['train']+=feasible*8
    scores,ranks=previous.tensors(rows,localpairs)
    opt=torch.optim.AdamW(model.parameters(),lr=.0001)
    scoreloader=DataLoader(TensorDataset(*scores['train']),batch_size=128,shuffle=True)
    rankloader=DataLoader(TensorDataset(*ranks['train']),batch_size=128,shuffle=True)
    outputs=[];history=[]
    for epoch in range(1,25):
        model.train()
        for x,b,y in scoreloader:
            opt.zero_grad();delta=weight*model(x);pred=rounded_score(b,delta,True)
            loss=4*nn.functional.mse_loss(torch.sigmoid(pred/400),torch.sigmoid(y/400))+.01*delta.square().mean()
            loss.backward();opt.step()
        for g,z,gb,zb,gf,zf,sign,cp in rankloader:
            opt.zero_grad();gap=sign*(rounded_score(gb,weight*gf*model(g),True)-rounded_score(zb,weight*zf*model(z),True))
            ref=sign*(gb-zb)
            loss=nn.functional.binary_cross_entropy_with_logits(gap/100,torch.sigmoid(cp.clamp(max=500)/100))+4*torch.where(ref>0,torch.relu(ref.clamp(max=10)-gap)/100,torch.zeros_like(gap)).mean()
            loss.backward();opt.step()
        if epoch in (8,16,24):
            saved=torch.load(OLD/f'weight-{int(weight*100)}.pt',map_location='cpu',weights_only=True)
            saved['state_dict']=copy.deepcopy(model.state_dict());saved['training_source']='search-correction-v29'
            path=ART/f'weight-{int(weight*100)}-epoch-{epoch}.pt';torch.save(saved,path);outputs.append(path)
            model.eval()
            with torch.inference_mode():
                g,z,gb,zb,gf,zf,sign,cp=ranks['train'];gap=sign*(rounded_score(gb,weight*gf*model(g))-rounded_score(zb,weight*zf*model(z)))
                history.append({'epoch':epoch,'training_pairs':len(gap),'training_correct':int((gap>0).sum())})
            print('Trained',weight,'epoch',epoch,'feasible focused pairs',len(feasible),flush=True)
    write(ART/f'training-{int(weight*100)}.json',{'feasible_focused_pairs':len(feasible),'history':history,'selection':'Fixed8/16/24 checkpoints selected by fresh searched development; not training rankings.'})
    return outputs


def pilot(checkpoint,weight):
    import subprocess,sys
    from dataclasses import asdict
    from benchmarks.match import play_game,summarize
    path=ROOT/'benchmarks/openings-search-correction-v29-pilot.json'
    subprocess.run([sys.executable,'-m','benchmarks.generate_strength_openings','--output',str(path),'--pairs','20','--seed','290030'],check=True)
    openings=json.loads(path.read_text());games=[];pgns='';planned=20
    hybrid=StrategicHybrid(checkpoint,weight);base=StrategicEvaluator(PRIORS)
    def selector(name,b,time_limit,depth_cap):
        p=search(b,depth=64,time_limit=time_limit,eval_fn=hybrid if name=='current' else base,use_lmr=True)
        stats=asdict(p);stats.pop('move');return p.move,stats
    for i,o in enumerate(openings,1):
        for color in chess.COLORS:
            record,pgn=play_game(o,color,.25,600,64,selector,i,'new_heuristic');games.append(record);pgns+=pgn
            summary=summarize(games);write(ART/'pilot/report.json',{'games':games,'summary':summary,'planned_games':planned,'status':'running'})
            (ART/'pilot/games.pgn').write_text(pgns);print('Pilot',len(games),summary,flush=True)
            if summary['errors'] or summary['unfinished'] or summary['interrupted']:raise RuntimeError('Incomplete pilot')
        if len(games)==20:
            if summary['score_fraction_completed']<.6:break
            planned=40
    write(ART/'pilot/report.json',{'games':games,'summary':summary,'planned_games':planned,'status':'completed'})


def main():
    ART.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(290029)
    write(ART/'protocol.json',{'architecture':'Unchanged892/64/32, quiet gate, raw250cp bound, blends10/25%; v28 warm start on same strategic baseline',
        'data':'19diagnosed development roots become training;40new heuristic fixed-node source games, whole first6ply families split24/8/8;64k quiet labels,256k consequential move confirmations, settled endpoint ranking feasibility; ordinary retention and correct-ranking protection.',
        'selection':'6fixed candidates (2weights x epochs8/16/24), choose by fresh searched depth3/250ms regret with no extra large errors, losing transitions or mate failures. Require untouched test gate before any20-game pilot; expand40 only if>=60%.',
        'bounds':'Endpoint attainability is a necessary capacity check, not proof of a root search repair. Captures/checks get no correction.',
        'family_amendment':'Canonical color/transposition-equivalent position after6plies owns one split; preliminary sequence-split collection archived before training.',
        'source_hashes':{p:digest(ROOT/p) for p in ('engine/strategic.py','engine/search.py','ml/run_search_correction_v29.py')}})
    with exclusive_cpu('v29 diagnosis search-data collection training and gates'):
        # Reuse the audited prior splits without rewriting old experiment outputs.
        original_art=previous.ART;previous.ART=ART
        rows,pairs,forbidden=previous.prepare();previous.ART=original_art
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            criticalrows,criticalpairs=diagnose(sf)
            criticalroots=[r['root'] for r in json.loads((OLD/'development.json').read_text())]
            roots,labels,newpairs=collect(sf,forbidden,criticalroots)
            mix,pairmix,focused=training_data(rows,pairs,roots,labels,newpairs,criticalrows,criticalpairs)
            candidates={}
            for w in (.1,.25):
                for path in train_candidates(w,mix,pairmix,focused):candidates[path.stem]=(path,w)
            models={'heuristic':StrategicEvaluator(PRIORS),**{n:StrategicHybrid(p,w) for n,(p,w) in candidates.items()}}
            previous.ART=ART
            development=previous.screen(sf,roots['val'],models,'development')
            eligible=[n for n in candidates if previous.qualifies(development,n)]
            selected=min(eligible,key=lambda n:development['250ms'][n]['mean_regret']) if eligible else None
            write(ART/'selection.json',{'selected':selected,'test_used_for_selection':False})
            if not selected:
                write(ART/'decision.json',{'qualifies_for_games':False,'reason':'No candidate passed fresh searched development','app_unchanged':True});return
            test=previous.screen(sf,roots['test'],{n:models[n] for n in ('heuristic',selected)},'test')
            passed=previous.qualifies(test,selected)
            write(ART/'decision.json',{'selected':selected,'qualifies_for_games':passed,'app_unchanged':True})
        if passed:
            # Reuse the established short conditional pilot with a distinct start file.
            pilot(*candidates[selected])

if __name__=='__main__':main()
