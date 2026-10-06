"""Controlled end-to-end input ablation, selected by searched development decisions."""
import copy,json,random
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import chess,chess.engine,torch,numpy as np
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from engine.search import search
from engine.strategic import StrategicEvaluator,PRIORS
from ml.collect_relationship_v35 import ROOT,ART,SPLITS
from ml.relationship_hybrid import RelationshipNet,RelationshipHybrid,metadata,features,ARMS
from ml.threat_model import THREAT_INPUT_SIZE
from ml.standpat_hybrid import StandPatHybrid
from ml.model import SCORE_SCALE
from ml.train import rounded_score
from ml.diagnose_critical_v23 import review
from ml.confirm_fatal_v23 import fatal
from ml.run_balanced_v19 import write,digest,SF
SNAPSHOTS=(8,16,24,32,40,48)

def prepare(d,arm):
    rs=d['rows']['train']+d['ordinary']['train'];byfen={r['fen']:r for r in d['rows']['train']}
    xs={r['fen']:torch.from_numpy(features(reconstruct(r),arm)) for r in rs}
    scores=TensorDataset(torch.stack([xs[r['fen']] for r in rs]),torch.tensor([r['strategic_base_cp'] for r in rs],dtype=torch.float32),torch.tensor([r['score_cp'] for r in rs],dtype=torch.float32))
    ps=d['pairs']['train']
    ranks=TensorDataset(torch.stack([xs[p['good_fen']] for p in ps]),torch.stack([xs[p['bad_fen']] for p in ps]),
        torch.tensor([byfen[p['good_fen']]['strategic_base_cp'] for p in ps],dtype=torch.float32),
        torch.tensor([byfen[p['bad_fen']]['strategic_base_cp'] for p in ps],dtype=torch.float32),
        torch.tensor([p['sign'] for p in ps],dtype=torch.float32),torch.tensor([2 if p['kind']=='repair' else 1 for p in ps],dtype=torch.float32))
    return scores,ranks

def static_stats(model,rows,pairs,arm):
    model.eval();xs=torch.from_numpy(np.stack([features(reconstruct(r),arm) for r in rows]))
    with torch.inference_mode():delta=.25*model(xs)*SCORE_SCALE
    predictions={r['fen']:round(r['strategic_base_cp']+delta[i].item()) for i,r in enumerate(rows)}
    repaired=damage=correct=0
    for p in pairs:
        ok=p['sign']*(predictions[p['good_fen']]-predictions[p['bad_fen']])>0
        correct+=ok;repaired+=ok and p['kind']=='repair';damage+=not ok and p['kind']=='retention'
    return {'pairs':len(pairs),'correct':correct,'repairs_correct':repaired,'retention_damage':damage,'near_bound_endpoints':int((delta.abs()>=60).sum().item())}


def fit(d):
    torch.set_num_threads(1);torch.manual_seed(350037)
    shared=RelationshipNet('current').state_dict();nn.init.zeros_(shared['net.4.weight']);nn.init.zeros_(shared['net.4.bias'])
    paths=[];history={}
    for arm in ARMS:
        torch.manual_seed(350037);model=RelationshipNet(arm);state=model.state_dict()
        for k,v in shared.items():
            if k=='net.0.weight' and arm=='relationships':state[k].zero_();state[k][:,:THREAT_INPUT_SIZE].copy_(v)
            else:state[k].copy_(v)
        model.load_state_dict(state)
        torch.save(metadata(copy.deepcopy(model.state_dict()),arm),ART/f'{arm}-zero.pt')
        scores,ranks=prepare(d,arm)
        scoreloader=DataLoader(scores,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(350038))
        rankloader=DataLoader(ranks,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(350039))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001);records=[]
        for epoch in range(1,49):
            model.train();losses=[]
            for x,b,y in scoreloader:
                optimizer.zero_grad();delta=.25*model(x)
                target=((y-b)/SCORE_SCALE).clamp(-45/SCORE_SCALE,45/SCORE_SCALE)
                loss=.1*nn.functional.smooth_l1_loss(delta,target,beta=.05);loss.backward();optimizer.step()
            for g,z,gb,zb,sign,importance in rankloader:
                optimizer.zero_grad();gd=.25*model(g);zd=.25*model(z)
                gap=sign*(rounded_score(gb,gd,True)-rounded_score(zb,zd,True));ref=sign*(gb-zb)
                target=torch.where(ref>0,ref.clamp(max=20),torch.minimum(torch.full_like(ref,20),ref+120))
                assert torch.all(target>0)
                loss=(importance*nn.functional.softplus((target-gap)/20)).mean()
                protection=torch.where(ref>0,torch.relu(target-gap)/20,torch.zeros_like(gap))
                loss=loss+2*protection.square().mean();loss.backward();optimizer.step();losses.append(loss.item())
            if epoch in SNAPSHOTS:
                path=ART/f'{arm}-epoch-{epoch}.pt';torch.save(metadata(copy.deepcopy(model.state_dict()),arm),path);paths.append(path)
                records.append({'epoch':epoch,'rank_loss':sum(losses)/len(losses),'train':static_stats(model,d['rows']['train'],d['pairs']['train'],arm),'validation':static_stats(model,d['rows']['val'],d['pairs']['val'],arm)});print('Trained',arm,records[-1],flush=True)
        history[arm]=records
    write(ART/'training.json',{'arms':history,'trained_hidden_layers':True,'shared_initialization':'From scratch same existing-input weights and layers; added input weights zero; no historical fitted weights/data.','architecture_parameters':{arm:sum(p.numel() for p in RelationshipNet(arm).parameters()) for arm in ARMS},'score_rows':len(d['rows']['train'])+len(d['ordinary']['train']),'ranking_pairs':len(d['pairs']['train'])})
    return paths

def audit(paths,d):
    rows=d['rows']['train'][:24]+d['rows']['val'][:24];results={}
    for path in paths+[ART/f'{a}-zero.pt' for a in ARMS]:
        adapter=RelationshipHybrid(path);maxerror=0;near=0
        for r in rows:
            b=reconstruct(r);before=b.fen();base=adapter.baseline(b)
            with torch.inference_mode():expected=round(base+.25*adapter.model(torch.from_numpy(features(b,adapter.arm))).item()*SCORE_SCALE)
            actual=adapter.evaluate_position(b);maxerror=max(maxerror,abs(actual-expected));near+=abs(actual-base)>=60
            assert maxerror<=1 and abs(actual-base)<=63 and actual==-adapter.evaluate_position(b.mirror()) and b.fen()==before
            if path.stem.endswith('zero'):assert actual==base
        results[path.name]={'positions':len(rows),'max_torch_runtime_difference_cp':maxerror,'symmetry_bound_board_checks':True,'near_bound_endpoints':near}
    write(ART/'inference-audit.json',results)

def summarize(records,names,mode):
    common=[r for r in records if all(r['probes'][n]['review']['cp_loss'] is not None for n in names)]
    return {n:{'mean_regret':sum(r['probes'][n]['review']['cp_loss'] for r in common)/len(common) if common else None,'comparable':len(common),
               'large_errors':sum((r['probes'][n]['review']['cp_loss'] or 0)>=150 for r in records),
               'losing_transitions':sum(fatal(r['probes'][n]['review']) for r in records),
               'allowed_mates':sum(r['probes'][n]['review']['allows_mate'] for r in records),
               'missed_mates':sum(r['probes'][n]['review']['missed_forced_mate'] for r in records),
               'mean_depth':sum(r['probes'][n]['depth'] for r in records)/len(records),
               'moves_changed_from_heuristic':sum(r['probes'][n]['move']!=r['probes']['heuristic']['move'] for r in records)} for n in names}

def screen(sf,roots,models,label,mode):
    path=ART/f'{label}.json'
    if path.exists():raise FileExistsError('Preserve frozen searched screen')
    records=[];options={'depth':3} if mode=='depth3' else {'depth':64,'time_limit':.25}
    for i,r in enumerate(roots):
        names=list(models);names=names[i%len(names):]+names[:i%len(names)];probes={};cache={}
        for name in names:
            p=search(reconstruct(r),eval_fn=models[name],use_lmr=True,**options)
            if mode=='depth3':assert p.depth==3 or abs(p.score or 0)>28000
            probes[name]={'move':p.move.uci(),'depth':p.depth,'nodes':p.nodes,'qnodes':p.qnodes,'elapsed':p.elapsed}
        for name,p in probes.items():
            if p['move'] not in cache:cache[p['move']]=review(sf,reconstruct(r),chess.Move.from_uci(p['move']),128000)
            p['review']=cache[p['move']]
        records.append({'root':r,'probes':probes});write(path,records)
        if (i+1)%4==0:print(label,i+1,'/',len(roots),flush=True)
    result=summarize(records,list(models),mode);write(ART/f'{label}-summary.json',result);return result

def nonworse(summary,name):
    a,b=summary[name],summary['heuristic']
    return a['comparable']>=20 and a['mean_regret'] is not None and a['mean_regret']<=b['mean_regret'] and all(a[k]<=b[k] for k in ('large_errors','losing_transitions','allowed_mates','missed_mates'))

def main():
    if not (ART/'collection.json').exists():raise FileNotFoundError('Complete collector first')
    if (ART/'training-protocol.json').exists():raise FileExistsError('Preserve completed/frozen training experiment')
    d=json.loads((ART/'collection.json').read_text());assert d['complete']
    counts={s:{'pairs':len(d['pairs'][s]),'repairs':sum(p['kind']=='repair' for p in d['pairs'][s])} for s in SPLITS}
    if counts['train']['pairs']<120 or counts['train']['repairs']<30 or min(counts[s]['pairs'] for s in ('val','test'))<25:
        write(ART/'coverage-before-training.json',{'passed':False,'counts':counts});raise ValueError('Insufficient independent decision coverage before training')
    write(ART/'training-protocol.json',{'arms':list(ARMS),'features_added':396,'hidden':[64,32],'raw_correction_bound_cp':250,'blend':.25,'gate':'noncheck_standpat_v2','epochs':48,'snapshots':SNAPSHOTS,'lr':.0005,'weight_decay':.001,'loss':'Softplus feasible ranking target20cp; repairimportance2; retention penalty2; supporting static clipped45cp lossweight0.1. No global correction-to-zero penalty.','initialization':'Both from scratch shared weights, augmented extra columns zero. Full hidden layers trainable, identical batch orders.','selection':'Each arm epoch selected by searched depth3 development: first no extra major errors/transitions/mates and nonworse mean loss; then mean loss; earliest epoch. All epochs evaluated; no static-only selection. Chosen arm snapshots frozen before test. Both selected snapshots get fixed-depth untouched test once, regardless of qualification, to report controlled representation comparison. Then development250ms; timed test only for candidates that pass both depth screens and timed development. No tuning after test; no automatic game match unless all gates pass.','collection_sha256':digest(ART/'collection.json'),'roots_sha256':digest(ART/'fresh-roots.json'),'source_sha256':{p:digest(ROOT/p) for p in ('ml/relationship_features.py','ml/relationship_hybrid.py','ml/train_relationship_v35.py','engine/search.py','engine/strategic.py')}})
    with exclusive_cpu('v35 controlled representations training and searched selection'):
        paths=fit(d);audit(paths,d);roots=json.loads((ART/'fresh-roots.json').read_text())
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
            models={'heuristic':StrategicEvaluator(PRIORS),**{p.stem:RelationshipHybrid(p) for p in paths}}
            development=screen(sf,roots['val'],models,'development-depth3','depth3');selected={}
            for arm in ARMS:
                options=[p for p in paths if p.stem.startswith(arm+'-')]
                def score(p):
                    s=development[p.stem]
                    return (not nonworse(development,p.stem),s['large_errors']+s['allowed_mates'] if not nonworse(development,p.stem) else 0,s['mean_regret'] if s['mean_regret'] is not None else float('inf'),int(p.stem.rsplit('-',1)[-1]))
                selected[arm]=min(options,key=score)
            write(ART/'selection.json',{'selected':{a:p.name for a,p in selected.items()},'selection_used_test':False})
            write(ART/'test-rankings.json',{a:static_stats(RelationshipHybrid(p).model,d['rows']['test'],d['pairs']['test'],a) for a,p in selected.items()})
            models={'heuristic':StrategicEvaluator(PRIORS),'old_v32':StandPatHybrid(ROOT/'ml/artifacts/standpat-v32/epoch-8.pt'),**{a:RelationshipHybrid(p) for a,p in selected.items()}}
            test=screen(sf,roots['test'],models,'test-depth3','depth3')
            timed=screen(sf,roots['val'],models,'development-250ms','250ms')
            eligible=[a for a,p in selected.items() if nonworse(development,p.stem) and nonworse(test,a) and test[a]['mean_regret']<test['heuristic']['mean_regret'] and nonworse(timed,a) and timed[a]['mean_regret']<timed['heuristic']['mean_regret']]
            timed_test=None
            if eligible:timed_test=screen(sf,roots['test'],{n:models[n] for n in ['heuristic','old_v32']+eligible},'test-250ms','250ms')
            qualified=[a for a in eligible if nonworse(timed_test,a) and timed_test[a]['mean_regret']<timed_test['heuristic']['mean_regret']]
            write(ART/'decision.json',{'selected':{a:p.name for a,p in selected.items()},'eligible_for_timed_test':eligible,'qualifies_for_games':qualified,'production':'unchanged heuristic','game_results':None})
            print('Decision',json.dumps(json.loads((ART/'decision.json').read_text())),flush=True)
if __name__=='__main__':main()
