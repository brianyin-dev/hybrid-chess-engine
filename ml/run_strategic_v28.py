"""Rebased residual training, development selection and untouched searched gate."""
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import chess
import chess.engine
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from benchmarks.match import play_game, summarize, opening_board
from engine.search import search
from engine.strategic import PRIORS, StrategicEvaluator
from ml.dataset import correction_factor
from ml.model import ChessNet, RELATIONAL_INPUT_SIZE, RELATIONAL_MODEL_VERSION, SCORE_SCALE, board_to_tensor
from ml.strategic_hybrid import StrategicHybrid
from ml.train import rounded_score
from ml.generate_search_data import key
from ml.run_balanced_v19 import readrows, write, digest, SF
from ml.run_search_v22 import exact
from ml.diagnose_critical_v23 import review
from ml.confirm_fatal_v23 import fatal

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'ml/artifacts/strategic-v28'
DATA=ROOT/'ml/data/balanced-v19-2026'


def prepare():
    baseline=StrategicEvaluator(PRIORS)
    rows={s:readrows(DATA/f'{s}.jsonl') for s in ('train','val','test')}
    pairs={s:readrows(DATA/f'pairs/{s}.jsonl') for s in rows}
    aliases={s:{key(r['fen']) for r in rs} for s,rs in rows.items()}
    games={s:{r['game_id'] for r in rs} for s,rs in rows.items()}
    for s,rs in pairs.items():
        for r in rs:
            aliases[s].update(key(r[f]) for f in ('good_fen','bad_fen','root_fen') if f in r)
            games[s].add(r['game_id'])
    # Historical data contain aliases shared across splits: exclude from training
    # and validation rather than claiming independence without checking it.
    forbidden=aliases['val']|aliases['test']
    rows['train']=[r for r in rows['train'] if key(r['fen']) not in forbidden and r['game_id'] not in games['val']|games['test']]
    pairs['train']=[r for r in pairs['train'] if not any(key(r[f]) in forbidden for f in ('good_fen','bad_fen','root_fen') if f in r) and r['game_id'] not in games['val']|games['test']]
    rows['val']=[r for r in rows['val'] if key(r['fen']) not in aliases['test'] and r['game_id'] not in games['test']]
    pairs['val']=[r for r in pairs['val'] if not any(key(r[f]) in aliases['test'] for f in ('good_fen','bad_fen','root_fen') if f in r) and r['game_id'] not in games['test']]
    extra=json.loads((ROOT/'benchmarks/results/strategy-v26/training-labels.json').read_text())
    rows['train'] += [{'fen':r['fen'],'score_cp':r['teacher_white_cp'],'game_id':2800000+r['source_game'],'source':'v26_quiet_search'} for r in extra if key(r['fen']) not in forbidden]
    manifests={}
    for split in rows:
        useful=[]
        for r in rows[split]:
            b=chess.Board(r['fen'])
            if b.is_game_over() or not correction_factor(b,1,True): continue
            r={**r,'strategic_base_cp':baseline(b)}
            r['strategic_residual_cp']=r['score_cp']-r['strategic_base_cp']
            useful.append(r)
        rows[split]=useful
        write(ART/f'rebased-{split}.json',useful)
        manifests[split]={'score_rows':len(useful),'ranking_pairs':len(pairs[split])}
    all_aliases={key(r['fen']) for rs in rows.values() for r in rs}
    for rs in pairs.values():
        all_aliases.update(key(r[f]) for r in rs for f in ('good_fen','bad_fen','root_fen') if f in r)
    write(ART/'data-manifest.json',{'counts':manifests,'source_manifest_sha256':digest(DATA/'manifest.json'),'rebase':'Stockfish label minus strategic-v26; old labels unchanged, no relabel compute','split_policy':'Existing game-separated splits, remove cross-split canonical aliases and game IDs. Legacy opening-family provenance incomplete; no global family-independence claim.'})
    return rows,pairs,all_aliases


def tensors(rows,pairs):
    base=StrategicEvaluator(PRIORS)
    scores={}
    ranks={}
    for s,rs in rows.items():
        scores[s]=(torch.stack([board_to_tensor(chess.Board(r['fen']),RELATIONAL_INPUT_SIZE) for r in rs]),
                   torch.tensor([r['strategic_base_cp'] for r in rs],dtype=torch.float32),
                   torch.tensor([r['score_cp'] for r in rs],dtype=torch.float32))
        arrays=[[] for _ in range(8)]
        for r in pairs[s]:
            good,bad=chess.Board(r['good_fen']),chess.Board(r['bad_fen'])
            values=(board_to_tensor(good,RELATIONAL_INPUT_SIZE),board_to_tensor(bad,RELATIONAL_INPUT_SIZE),
                    float(base(good)),float(base(bad)),float(correction_factor(good,1,True)),
                    float(correction_factor(bad,1,True)),float(r['sign']),float(r['cp_loss']))
            for a,v in zip(arrays,values):a.append(v)
        ranks[s]=tuple(torch.stack(a) if i<2 else torch.tensor(a,dtype=torch.float32) for i,a in enumerate(arrays))
    return scores,ranks


def train(weight,scores,ranks):
    torch.manual_seed(280028)
    model=ChessNet(RELATIONAL_INPUT_SIZE,250,True,(64,32))
    # Old corrections target another baseline: do not reuse them unchanged.
    nn.init.zeros_(model.net[4].weight);nn.init.zeros_(model.net[4].bias)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005)
    scoreloader=DataLoader(TensorDataset(*scores['train']),batch_size=128,shuffle=True)
    pairloader=DataLoader(TensorDataset(*ranks['train']),batch_size=128,shuffle=True)
    history=[];best=None;state=None;epoch_selected=None
    def validation():
        model.eval()
        with torch.inference_mode():
            x,b,y=scores['val'];pred=rounded_score(b,weight*model(x))
            mse=nn.functional.mse_loss(torch.sigmoid(pred/400),torch.sigmoid(y/400)).item()
            g,z,gb,zb,gf,zf,sign,cp=ranks['val']
            gap=sign*(rounded_score(gb,weight*gf*model(g))-rounded_score(zb,weight*zf*model(z)))
            ref=sign*(gb-zb);correct=int((gap>0).sum());regress=int(((ref>0)&(gap<=0)).sum())
        return {'correct':correct,'regressions':regress,'pairs':len(gap),'score_proxy_mse':mse},(-correct+2*regress,mse)
    for epoch in range(1,25):
        model.train()
        for x,b,y in scoreloader:
            opt.zero_grad();correction=weight*model(x)
            pred=rounded_score(b,correction,True)
            loss=4*nn.functional.mse_loss(torch.sigmoid(pred/400),torch.sigmoid(y/400))+.01*correction.square().mean()
            loss.backward();opt.step()
        for g,z,gb,zb,gf,zf,sign,cp in pairloader:
            opt.zero_grad()
            gap=sign*(rounded_score(gb,weight*gf*model(g),True)-rounded_score(zb,weight*zf*model(z),True))
            ref=sign*(gb-zb)
            ranking=nn.functional.binary_cross_entropy_with_logits(gap/100,torch.sigmoid(cp.clamp(max=500)/100))
            protection=torch.where(ref>0,torch.relu(ref.clamp(max=10)-gap)/100,torch.zeros_like(gap)).mean()
            (ranking+4*protection).backward();opt.step()
        val,k=validation();history.append({'epoch':epoch,**val})
        if best is None or k<best:best=k;state=copy.deepcopy(model.state_dict());epoch_selected=epoch
        if epoch%8==0:print('Training',weight,'epoch',epoch,val,flush=True)
    model.load_state_dict(state)
    path=ART/f'weight-{int(weight*100)}.pt'
    torch.save({'version':RELATIONAL_MODEL_VERSION,'input_size':RELATIONAL_INPUT_SIZE,'score_scale':SCORE_SCALE,
                'target_mode':'residual','correction_limit_cp':250,'color_consistent':True,'hidden_sizes':[64,32],
                'baseline_id':'strategic-v26','baseline_weights':list(PRIORS),'training_correction_weight':weight,
                'training_quiet_only':True,'state_dict':state},path)
    write(ART/f'training-{int(weight*100)}.json',{'history':history,'best_epoch':epoch_selected,'objective':'Exact quiet gate, full hybrid integer rounding with straight-through gradient; teacher score proxy plus settled move rankings; protect new-heuristic correct rankings; bounded raw250cp correction. No old-model warm start.'})
    # Compare actual optimized runtime to Torch training predictions.
    adapter=StrategicHybrid(path,weight);base=StrategicEvaluator(PRIORS)
    with torch.inference_mode():
        for r in json.loads((ART/'rebased-val.json').read_text())[:100]:
            b=chess.Board(r['fen']);expected=round(base(b)+weight*model(board_to_tensor(b,RELATIONAL_INPUT_SIZE).unsqueeze(0)).item()*SCORE_SCALE)
            assert abs(adapter.evaluate_position(b)-expected)<=1
            assert adapter(b)==-adapter(b.mirror())
    return path


def roots(sf,forbidden):
    path=ART/'searched-roots.json'
    if path.exists():return json.loads(path.read_text())
    openings=json.loads((ROOT/'benchmarks/openings-hybrid-v28-data.json').read_text())
    families={}
    for o in openings:families.setdefault(tuple(o['moves'][:4]),[]).append(o)
    groups=sorted(families.items(),key=lambda v:(-len(v[1]),v[0]));assigned={'val':[],'test':[]}
    for family,items in groups:
        split=min(assigned,key=lambda s:len(assigned[s]));assigned[split]+=items
    result={s:[] for s in assigned}
    for split,items in assigned.items():
        for i,o in enumerate(items[:20]):
            b=opening_board(o)
            for _ in range(24+i%12):
                if b.is_game_over():break
                b.push(exact(sf,b,nodes=2000)['pv'][0])
            if b.is_game_over() or key(b.fen()) in forbidden:continue
            result[split].append({'fen':b.fen(),'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack],'opening':o,'family':o['moves'][:4]})
    assert min(map(len,result.values()))>=16
    assert not {tuple(r['family']) for r in result['val']}&{tuple(r['family']) for r in result['test']}
    write(path,result)
    return result


def screen(sf,rs,models,label):
    records=[]
    for i,r in enumerate(rs):
        probes={};cache={};names=list(models);names=names[i%len(names):]+names[:i%len(names)]
        for mode,opts in [('depth3',{'depth':3}),('250ms',{'depth':64,'time_limit':.25})]:
            ps={}
            for name in names:
                p=search(reconstruct(r),eval_fn=models[name],use_lmr=True,**opts)
                ps[name]={'move':p.move.uci(),'depth':p.depth,'elapsed':p.elapsed}
            for name,p in ps.items():
                if p['move'] not in cache:cache[p['move']]=review(sf,reconstruct(r),chess.Move.from_uci(p['move']),256000)
                p['review']=cache[p['move']]
            probes[mode]=ps
        records.append({'root':r,'probes':probes});write(ART/f'{label}.json',records)
        if (i+1)%5==0:print(label,i+1,'/',len(rs),flush=True)
    summary={}
    for mode in ('depth3','250ms'):
        common=[r for r in records if all(r['probes'][mode][n]['review']['cp_loss'] is not None for n in models)]
        summary[mode]={n:{'mean_regret':sum(r['probes'][mode][n]['review']['cp_loss'] for r in common)/len(common) if common else float('inf'),
            'comparable':len(common),'large_errors':sum((r['probes'][mode][n]['review']['cp_loss'] or 0)>=150 for r in records),
            'losing_transitions':sum(fatal(r['probes'][mode][n]['review']) for r in records),
            'allowed_mates':sum(r['probes'][mode][n]['review']['allows_mate'] for r in records)} for n in models}
    write(ART/f'{label}-summary.json',summary)
    return summary


def qualifies(summary,name):
    for mode,rs in summary.items():
        a,b=rs[name],rs['heuristic']
        if a['comparable']<12 or a['mean_regret']>b['mean_regret']:return False
        if any(a[k]>b[k] for k in ('large_errors','losing_transitions','allowed_mates')):return False
    return summary['250ms'][name]['mean_regret']<summary['250ms']['heuristic']['mean_regret']


def pilot(cp,weight):
    import subprocess,sys
    path=ROOT/'benchmarks/openings-hybrid-v28-pilot.json'
    subprocess.run([sys.executable,'-m','benchmarks.generate_strength_openings','--output',str(path),'--pairs','20','--seed','280029'],check=True)
    openings=json.loads(path.read_text());records=[];pgns=''
    hybrid=StrategicHybrid(cp,weight);baseline=StrategicEvaluator(PRIORS)
    def selector(name,b,time_limit,depth_cap):
        p=search(b,eval_fn=hybrid if name=='current' else baseline,depth=64,time_limit=time_limit,use_lmr=True)
        stats=asdict(p);stats.pop('move');return p.move,stats
    planned=20
    for i,o in enumerate(openings,1):
        for color in chess.COLORS:
            g,pgn=play_game(o,color,.25,600,64,selector,i,'new_heuristic');records.append(g);pgns+=pgn
            s=summarize(records);write(ART/'pilot/report.json',{'games':records,'summary':s,'planned_games':planned,'status':'running'})
            (ART/'pilot/games.pgn').write_text(pgns);print('Pilot',len(records),s['wins'],'W',s['draws'],'D',s['losses'],'L',flush=True)
            if s['errors'] or s['unfinished'] or s['interrupted']:raise ValueError('Incomplete pilot game')
        if len(records)==20:
            if s['score_fraction_completed']<.6:break
            planned=40
    write(ART/'pilot/report.json',{'games':records,'summary':s,'planned_games':planned,'status':'completed','expanded':planned==40})


def main():
    ART.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    write(ART/'protocol.json',{'weights':[.1,.25],'training_epochs':24,'architecture':'892/64/32 color-consistent residual, raw250cp bound','selection':'Epochs by existing heldout rounded ranking protection; blend by fresh searched development. Test once. Require nonworse depth3, strictly lower timed common-position regret, no extra large errors, losing transitions or mates. Pilot20 games at250ms, expand40 only if >=60%.','app':'Strategic heuristic only; hybrid remains experimental until game improvement.','source_hashes':{p:digest(ROOT/p) for p in ('engine/strategic.py','engine/search.py','ml/strategic_hybrid.py','ml/run_strategic_v28.py')}})
    with exclusive_cpu('v28 rebase train searched gate and conditional pilot'):
        rows,pairs,forbidden=prepare();scores,ranks=tensors(rows,pairs)
        paths={w:train(w,scores,ranks) for w in (.1,.25)}
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            rs=roots(sf,forbidden)
            models={'heuristic':StrategicEvaluator(PRIORS),**{str(w):StrategicHybrid(p,w) for w,p in paths.items()}}
            development=screen(sf,rs['val'],models,'development')
            eligible=[str(w) for w in paths if qualifies(development,str(w))]
            selected=min(eligible,key=lambda n:development['250ms'][n]['mean_regret']) if eligible else None
            write(ART/'selection.json',{'selected':selected,'selection_uses_test':False})
            if selected is None:
                write(ART/'decision.json',{'qualifies_for_games':False,'reason':'Neither blend passed searched development; no test selection or games','app_uses':'strategic heuristic only'});return
            test=screen(sf,rs['test'],{n:models[n] for n in ('heuristic',selected)},'test')
            passed=qualifies(test,selected)
            write(ART/'decision.json',{'selected':selected,'qualifies_for_games':passed,'app_uses':'strategic heuristic only'})
        if passed:pilot(paths[float(selected)],float(selected))

if __name__=='__main__':main()
