"""Broad, split-isolated settled rankings and one fixed-architecture training run."""
import copy
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

import chess
import chess.engine
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from benchmarks.match import opening_board
from engine.search import search
from engine.strategic import PRIORS,StrategicEvaluator
from ml import run_strategic_v28 as common
from ml import run_search_correction_v29 as prior
from ml.dataset import correction_factor
from ml.generate_search_data import key
from ml.model import SCORE_SCALE
from ml.strategic_hybrid import StrategicHybrid
from ml.train import rounded_score
from ml.run_balanced_v19 import write,digest,SF,readrows
from ml.run_search_v22 import exact
from ml.diagnose_critical_v23 import review

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/ranking-v30'
INITIAL=ROOT/'ml/artifacts/search-correction-v29/weight-25-epoch-16.pt'
WEIGHT=.25
TARGETS={'train':300,'val':60,'test':60}


def pair_key(good,bad):
    """Canonical good/bad endpoints, independent of color-mirror orientation."""
    return key(good),key(bad)


def phase(b):
    from engine.evaluation import _phase
    return 'endgame' if _phase(b)<=10 else 'middlegame'


def allocation():
    path=ART/'allocation.json'
    if path.exists():return json.loads(path.read_text())
    openings=json.loads((ROOT/'benchmarks/openings-ranking-v30.json').read_text())
    groups=defaultdict(list)
    for o in openings:groups[prior.family(o['moves'])].append(o)
    assigned={s:[] for s in TARGETS};ratio={'train':3,'val':1,'test':1}
    for f,items in sorted(groups.items(),key=lambda item:(-len(item[1]),item[0])):
        split=min(assigned,key=lambda s:len(assigned[s])/ratio[s])
        assigned[split].extend({**o,'family':f} for o in items)
    rng=random.Random(300030)
    for items in assigned.values():rng.shuffle(items)
    assert all(len(items)>=20 for items in assigned.values())
    write(path,assigned)
    return assigned


def reserved_aliases():
    result=set()
    for p in (ROOT/'ml/data').glob('**/*.jsonl'):
        for r in readrows(p):
            for f in ('fen','good_fen','bad_fen','root_fen'):
                if f in r:result.add(key(r[f]))
    for folder in ('strategic-v28','search-correction-v29'):
        for p in (ROOT/f'ml/artifacts/{folder}').glob('rebased-*.json'):
            result.update(key(r['fen']) for r in json.loads(p.read_text()))
        for filename in ('development.json','test.json'):
            p=ROOT/f'ml/artifacts/{folder}/{filename}'
            if p.exists():result.update(key(r['root']['fen']) for r in json.loads(p.read_text()))
    return result


def collect(sf,assigned):
    statepath=ART/'collection.json'
    data=json.loads(statepath.read_text()) if statepath.exists() else {
        'pairs':{s:[] for s in TARGETS},'rows':{s:[] for s in TARGETS},'roots':{s:[] for s in TARGETS},'sources':[],
        'statistics':{'sampled_roots':0,'disagreements':0,'attempted_confirmations':0,'rejections':{}},'complete':False}
    if data['complete']:return data
    reserved=reserved_aliases();owners={};used_endpoints=set();seenpairs=set()
    for split,ps in data['pairs'].items():
        for p in ps:
            seenpairs.add(pair_key(p['good_fen'],p['bad_fen']))
            for f in ('good_fen','bad_fen','root_fen'):owners[key(p[f])]=split
            used_endpoints.update((key(p['good_fen']),key(p['bad_fen'])))
    for split,rs in data['roots'].items():
        for r in rs:owners[key(r['fen'])]=split
    baseline=StrategicEvaluator(PRIORS);neural=StrategicHybrid(INITIAL,WEIGHT)
    rng=random.Random(300031)
    if 'rng_state' in data:
        state=data['rng_state'];rng.setstate((state[0],tuple(state[1]),state[2]))
    rejects=Counter(data['statistics']['rejections'])
    def reject(reason):rejects[reason]+=1
    def settle(root,move):
        # Existing strict gate settling retains full history and 64k endpoint labels.
        return prior.settle(sf,root,move)
    for split,items in assigned.items():
        repairs=sum(p['kind']=='repair' for p in data['pairs'][split])
        retention=sum(p['kind']=='retention' for p in data['pairs'][split])
        wanted_repairs=30 if split=='train' else 6
        for gi,o in enumerate(items):
            gid=3000000+list(TARGETS).index(split)*10000+gi
            if any(g['game_id']==gid for g in data['sources']):continue
            if len(data['pairs'][split])>=TARGETS[split] and repairs>=wanted_repairs:break
            b=opening_board(o);accepted_slots=set();local_pairs=[];local_rows=[];local_roots=[]
            max_pairs=5
            source_policy=('stockfish_selfplay_3000nodes','heuristic_selfplay','heuristic_vs_stockfish_2000nodes')[gi%3]
            offset=gi%2
            source_white=(gi%2==0)
            for ply in range(160):
                if b.is_game_over():break
                root={'fen':b.fen(),'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack],
                      'game_id':gid,'family':o['family'],'split':split,'source_ply':ply}
                alias=key(b.fen())
                if split!='train' and ply in (16+offset,80+offset) and len(data['roots'][split])+len(local_roots)<24:
                    if alias not in reserved and owners.get(alias,split)==split:
                        local_roots.append(root);owners[alias]=split
                # At most one accepted pair in each separated 24-ply window.
                slot=(ply-16)//24
                collecting=len(data['pairs'][split])+len(local_pairs)<TARGETS[split]
                if collecting and ply>=16+offset and (ply-offset)%8==0 and slot not in accepted_slots and len(local_pairs)<max_pairs and alias not in reserved and owners.get(alias,split)==split:
                    data['statistics']['sampled_roots']+=1
                    h=search(b,depth=2,node_limit=2500,eval_fn=baseline,use_lmr=True)
                    n=search(b,depth=2,node_limit=2500,eval_fn=neural,use_lmr=True)
                    disagreement=h.move!=n.move
                    data['statistics']['disagreements']+=int(disagreement)
                    infos=sf.analyse(b,chess.engine.Limit(nodes=8000),multipv=min(4,b.legal_moves.count()),game=object())
                    if infos and infos[0].get('pv'):
                        best=infos[0]['pv'][0]
                        alternatives=list(dict.fromkeys([n.move,h.move]+[info['pv'][0] for info in infos[1:] if info.get('pv')]))
                        # Quiet diverse alternatives are fallback retention anchors,
                        # never accepted without root and endpoint teacher confirmation.
                        quiet_moves=[m for m in b.legal_moves if not b.is_capture(m) and not m.promotion and m!=best]
                        rng.shuffle(quiet_moves);alternatives+=quiet_moves[:2]
                        for bad in alternatives[:6]:
                            if bad is None or bad==best:continue
                            data['statistics']['attempted_confirmations']+=1
                            cheap=review(sf,b,bad,16000)
                            if cheap['cp_loss'] is None or cheap['cp_loss']<15 or abs(cheap['best_score']['cp'] or 0)>1200:
                                reject('preliminary_gap');continue
                            nodes=256000 if disagreement and bad in (n.move,h.move) else 128000
                            confirmed=review(sf,b,bad,nodes)
                            if confirmed['cp_loss'] is None or confirmed['cp_loss']<15:
                                reject('unconfirmed_root_ranking');continue
                            good_move=chess.Move.from_uci(confirmed['best_move'])
                            g,z=settle(root,good_move),settle(root,bad)
                            if not g or not z:
                                reject('not_quiet_or_mate');continue
                            sign=1 if b.turn else -1;gap=sign*(g['score_cp']-z['score_cp'])
                            if gap<15 or max(abs(g['score_cp']),abs(z['score_cp']))>1200:
                                reject('settled_gap');continue
                            bounds=prior.interval(g,z,sign,WEIGHT)
                            if not bounds['permits_correct_ranking']:
                                reject('bounded_correction_infeasible');continue
                            keys=pair_key(g['fen'],z['fen'])
                            if keys[0]==keys[1] or keys in seenpairs or any(k in used_endpoints or k in reserved or owners.get(k,split)!=split for k in keys):
                                reject('duplicate_or_reserved_endpoint');continue
                            kind='repair' if bounds['base_margin_cp']<=0 else 'retention'
                            # Reserve 10% of every split for genuinely wrong heuristic rankings.
                            if kind=='retention' and retention+sum(p['kind']=='retention' for p in local_pairs)>=TARGETS[split]-wanted_repairs:
                                reject('retention_quota');continue
                            if kind=='retention' and bounds['base_margin_cp']>150:
                                reject('trivial_retention');continue
                            p={'good_fen':g['fen'],'bad_fen':z['fen'],'root_fen':b.fen(),'sign':sign,'cp_loss':gap,
                               'game_id':gid,'family':o['family'],'source_ply':ply,'kind':kind,'stage':phase(b),
                               'good_move':good_move.uci(),'bad_move':bad.uci(),'nn_move':n.move.uci(),'heuristic_move':h.move.uci(),
                               'disagreement':disagreement,'root_review':confirmed,'root_teacher_nodes':nodes,
                               'capacity':bounds,'good_history':g['history'],'bad_history':z['history'],'initial_fen':root['initial_fen']}
                            local_pairs.append(p);local_rows.extend({**leaf,'game_id':gid,'family':o['family'],'source':'v30_settled_ranking','kind':kind} for leaf in (g,z))
                            accepted_slots.add(slot);seenpairs.add(keys);used_endpoints.update(keys)
                            owners[alias]=split
                            for k in keys:owners[k]=split
                            break
                # Source policy is fixed independently of pair acceptance/outcomes.
                if source_policy=='stockfish_selfplay_3000nodes':
                    move=exact(sf,b,nodes=3000)['pv'][0]
                elif source_policy=='heuristic_vs_stockfish_2000nodes' and b.turn!=source_white:
                    move=exact(sf,b,nodes=2000)['pv'][0]
                else:
                    move=search(b,depth=64,node_limit=1500,eval_fn=baseline,use_lmr=True).move
                if move is None:break
                b.push(move)
            data['pairs'][split]+=local_pairs;data['rows'][split]+=local_rows;data['roots'][split]+=local_roots
            repairs+=sum(p['kind']=='repair' for p in local_pairs);retention+=sum(p['kind']=='retention' for p in local_pairs)
            data['sources'].append({'game_id':gid,'split':split,'family':o['family'],'opening':o,'policy':source_policy,
                'history':[m.uci() for m in b.move_stack],'final_fen':b.fen(),'pairs':len(local_pairs),'truncated':not b.is_game_over()})
            data['statistics']['rejections']=dict(rejects);data['rng_state']=rng.getstate();write(statepath,data)
            print('Collect',split,'sources',sum(g['split']==split for g in data['sources']),'pairs',len(data['pairs'][split]),'repair',repairs,'retention',retention,flush=True)
        if len(data['pairs'][split])<TARGETS[split]:
            write(ART/'collection-shortfall.json',{'split':split,'requested':TARGETS[split],'actual':len(data['pairs'][split]),'repair':repairs})
            raise RuntimeError('Frozen source pool exhausted before pair quota; preserve data, do not fit a tiny targeted set')
    data['complete']=True;write(statepath,data)
    return data


def mix(data):
    base=StrategicEvaluator(PRIORS);rows=copy.deepcopy(data['rows']);pairs=copy.deepcopy(data['pairs'])
    heldout=set()
    for s in ('val','test'):
        heldout.update(key(r['fen']) for r in data['roots'][s])
        heldout.update(key(p[f]) for p in pairs[s] for f in ('good_fen','bad_fen','root_fen'))
    rng=random.Random(300032)
    legacy=[r for r in json.loads((ROOT/'ml/artifacts/strategic-v28/rebased-train.json').read_text()) if key(r['fen']) not in heldout]
    rng.shuffle(legacy);retention=legacy[:2000]
    for s in rows:
        for r in rows[s]:
            r['strategic_base_cp']=base(chess.Board(r['fen']));r['strategic_residual_cp']=r['score_cp']-r['strategic_base_cp']
    rows['train']=retention+rows['train']*3
    write(ART/'retained-score-rows.json',retention)
    write(ART/'training-data.json',{'score_counts':{s:len(rs) for s,rs in rows.items()},'pair_counts':{s:len(ps) for s,ps in pairs.items()},
        'ordinary_retention':len(retention),'new_score_sampling_multiplier':3,'repair_pair_loss_weight':3,
        'limits':'Whole fresh families/games isolated. Legacy family provenance incomplete. Endpoint bounds permit but do not prove searched repair.'})
    return rows,pairs


def train(rows,pairs):
    torch.manual_seed(300033);model=StrategicHybrid(INITIAL,WEIGHT).neural.model
    scores,ranks=common.tensors(rows,pairs)
    opt=torch.optim.AdamW(model.parameters(),lr=.0001)
    scoreloader=DataLoader(TensorDataset(*scores['train']),batch_size=128,shuffle=True)
    importance=torch.tensor([3. if p['kind']=='repair' else 1. for p in pairs['train']])
    rankloader=DataLoader(TensorDataset(*ranks['train'],importance),batch_size=128,shuffle=True)
    paths=[];history=[]
    for epoch in range(1,25):
        model.train()
        for x,b,y in scoreloader:
            opt.zero_grad();delta=WEIGHT*model(x);pred=rounded_score(b,delta,True)
            loss=4*nn.functional.mse_loss(torch.sigmoid(pred/400),torch.sigmoid(y/400))+.01*delta.square().mean()
            loss.backward();opt.step()
        for g,z,gb,zb,gf,zf,sign,cp,importance in rankloader:
            opt.zero_grad();gap=sign*(rounded_score(gb,WEIGHT*gf*model(g),True)-rounded_score(zb,WEIGHT*zf*model(z),True))
            ref=sign*(gb-zb)
            ranking=nn.functional.binary_cross_entropy_with_logits(gap/100,torch.sigmoid(cp.clamp(max=500)/100),reduction='none')
            protection=torch.where(ref>0,torch.relu(ref.clamp(max=10)-gap)/100,torch.zeros_like(gap))
            ((ranking*importance).mean()+4*protection.mean()).backward();opt.step()
        if epoch in (8,16,24):
            path=ART/f'epoch-{epoch}.pt';saved=torch.load(INITIAL,map_location='cpu',weights_only=True)
            saved.update(state_dict=copy.deepcopy(model.state_dict()),training_source='ranking-v30');torch.save(saved,path);paths.append(path)
            model.eval()
            with torch.inference_mode():
                g,z,gb,zb,gf,zf,sign,cp=ranks['train'];gap=sign*(rounded_score(gb,WEIGHT*gf*model(g))-rounded_score(zb,WEIGHT*zf*model(z)))
                repairs=torch.tensor([p['kind']=='repair' for p in pairs['train']]);history.append({'epoch':epoch,'correct':int((gap>0).sum()),'total':len(gap),'repair_correct':int(((gap>0)&repairs).sum()),'repair_total':int(repairs.sum())})
            print('Training epoch',epoch,history[-1],flush=True)
    write(ART/'training.json',{'history':history,'single_training_run':True,'architecture_changed':False,'blend':WEIGHT,'selection':'Fixed8/16/24snapshots by fresh searched validation; no test tuning.'})
    return paths


def gate(summary,name):
    return common.qualifies(summary,name) and all(rs[name]['comparable']>=20 for rs in summary.values())


def main():
    ART.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    protocol={'targets':TARGETS,'weight':WEIGHT,'single_training_run':True,'initial_checkpoint_sha256':digest(INITIAL),
        'architecture':'Unchanged892/64/32, raw250cp bound, quiet gate, rounded25%correction',
        'source':'320frozen starts; whole canonical6ply families split3:1:1; up to160sourceplies, heuristic1500nodes; thirds heuristic selfplay, heuristic-vs-SF2000nodes, SFselfplay3000nodes; balanced side-to-move offsets; five accepted separated-window pairs/game maximum.',
        'labels':'16k preliminary,128k root confirmation or256k for consequential NN/heuristic disagreements; strict quiet settling8k/step,max12steps;64k endpoint labels;>=15cp gaps; reserved aliases and reused endpoints excluded; bounded feasible;10% minimum repair pairs.',
        'selection':'One training run24epochs; fixed8/16/24snapshots searched on24fresh validation roots atdepth3/250ms; common>=20, no extra large mistakes/losing transitions/mates, strictly better timed mean. One selected candidate tested once on24fresh test roots. Pilot20 then40onlyif>=60%.',
        'collection_amendment':'Before any fitting, archived preliminary7sources/6pairs; corrected even-ply White bias and broadened source policies because short tactical collapses yielded sparse middlegame/endgame coverage. No candidate validation/test outcomes used.',
        'source_hashes':{p:digest(ROOT/p) for p in ('engine/strategic.py','engine/search.py','ml/run_ranking_v30.py')}}
    p=ART/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Frozen protocol/source changed'
    else:write(p,protocol)
    assigned=allocation()
    with exclusive_cpu('v30 broad ranking collection single training and unseen gate'):
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32});data=collect(sf,assigned)
            for s in ('val','test'):assert len(data['roots'][s])==24,'Insufficient predeclared searched roots'
            rows,pairs=mix(data);paths=train(rows,pairs)
            common.ART=ART
            models={'heuristic':StrategicEvaluator(PRIORS),'previous_nn':StrategicHybrid(INITIAL,WEIGHT),**{p.stem:StrategicHybrid(p,WEIGHT) for p in paths}}
            development=common.screen(sf,data['roots']['val'],models,'development')
            eligible=[p.stem for p in paths if gate(development,p.stem)]
            selected=min(eligible,key=lambda n:development['250ms'][n]['mean_regret']) if eligible else None
            write(ART/'selection.json',{'selected':selected,'test_used_for_selection':False})
            if not selected:
                write(ART/'decision.json',{'qualifies_for_games':False,'reason':'No snapshot passed searched validation','app_unchanged':True});return
            test=common.screen(sf,data['roots']['test'],{n:models[n] for n in ('heuristic',selected)},'test')
            passed=gate(test,selected);write(ART/'decision.json',{'selected':selected,'qualifies_for_games':passed,'app_unchanged':True})
        if passed:pilot(ART/f'{selected}.pt')


def pilot(path):
    # Reuse the audited pilot mechanics with distinct artifact/opening paths.
    import subprocess,sys
    from dataclasses import asdict
    from benchmarks.match import play_game,summarize
    opening_path=ROOT/'benchmarks/openings-ranking-v30-pilot.json'
    subprocess.run([sys.executable,'-m','benchmarks.generate_strength_openings','--output',str(opening_path),'--pairs','20','--seed','300034'],check=True)
    openings=json.loads(opening_path.read_text());games=[];pgns='';planned=20
    hybrid=StrategicHybrid(path,WEIGHT);baseline=StrategicEvaluator(PRIORS)
    def selector(name,b,time_limit,depth_cap):
        p=search(b,depth=64,time_limit=time_limit,eval_fn=hybrid if name=='current' else baseline,use_lmr=True)
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

if __name__=='__main__':main()
