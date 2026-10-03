"""Profile, fit general features and select by searched development choices."""
import cProfile,json,pstats,random
from collections import Counter
from statistics import median
from time import perf_counter
import chess,chess.engine,torch
from benchmarks.build_strategy_v26 import ROOT,ART,save,reconstruct,CATEGORIES
from benchmarks import evaluation_baseline_v26 as frozen
from engine.evaluation import evaluate as native
from engine.strategic import StrategicEvaluator,PRIORS,BOUNDS,FEATURE_NAMES
from engine.search import search
from benchmarks.cpu_lock import exclusive_cpu
from ml.diagnose_critical_v23 import SF,review
from ml.confirm_fatal_v23 import fatal
from ml.run_search_v22 import exact
from ml.generate_search_data import key

class Collector:
    cacheable_by_fen=True
    def __init__(self,rng):self.model=StrategicEvaluator();self.rng=rng;self.rows=[];self.count=0
    def evaluate_position(self,b):
        if not b.is_check() and next(b.generate_legal_captures(),None) is None:
            self.count+=1;r={'fen':b.fen(),'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack]}
            if len(self.rows)<16:self.rows.append(r)
            else:
                i=self.rng.randrange(self.count)
                if i<16:self.rows[i]=r
        return self.model(b)
    __call__=evaluate_position

def profiling(roots):
    positions=[next(r for r in roots if r['category']==cat) for cat in CATEGORIES]
    trials=[]
    for r in positions:
        pair={}
        for name,model in [('native',native),('cached_zero',StrategicEvaluator())]:
            times=[];results=[]
            for _ in range(3):
                t=perf_counter();p=search(reconstruct(r),depth=3,eval_fn=model,use_lmr=True);times.append(perf_counter()-t)
                results.append((p.move.uci(),p.score,p.nodes,p.qnodes))
            pair[name]={'median_seconds':median(times),'result':results[0],'times':times}
            assert len(set(results))==1
        assert pair['native']['result']==pair['cached_zero']['result']
        trials.append({'category':r['category'],'trials':pair})
    pr=cProfile.Profile();pr.enable()
    for r in positions:search(reconstruct(r),depth=3,use_lmr=True)
    pr.disable();pr.dump_stats(str(ART/'native-search.prof'));stats=pstats.Stats(pr)
    top=[{'file':k[0],'function':k[2],'calls':v[1],'self_seconds':v[2],'cumulative_seconds':v[3]} for k,v in sorted(stats.stats.items(),key=lambda x:x[1][3],reverse=True)[:25]]
    a=sum(r['trials']['native']['median_seconds'] for r in trials);z=sum(r['trials']['cached_zero']['median_seconds'] for r in trials)
    save('profile.json',{'fixed_depth':3,'repeats':3,'positions':trials,'summed_median_seconds_native':a,'summed_median_seconds_cached_zero':z,'reduction_fraction':1-z/a,'exact_move_score_node_parity':True,'top_functions':top})

def collect_and_fit(sf,roots,validation,test):
    rng=random.Random(260030);forbidden={key(r['fen']) for r in validation+test};seen=set()
    rows=json.loads((ART/'training-labels.json').read_text()) if (ART/'training-labels.json').exists() else []
    model=StrategicEvaluator(PRIORS)
    for i,r in enumerate([] if rows else roots):
        collector=Collector(rng);search(reconstruct(r),depth=2,node_limit=2500,eval_fn=collector,use_lmr=True)
        rng.shuffle(collector.rows);n=0
        for leaf in collector.rows:
            alias=key(leaf['fen'])
            if alias in seen or alias in forbidden:continue
            b=reconstruct(leaf);info=exact(sf,b,nodes=64000);cp=info['score'].white().score()
            if cp is None:continue
            seen.add(alias);base,x=model.analyze(b)
            rows.append({**leaf,'family':r['family'],'source_game':r['source_game'],'category':r['category'],'root_fen':r['fen'],'teacher_white_cp':cp,'base_white_cp':base,'features':x,'teacher_nodes':64000});n+=1
            if n==4:break
        if (i+1)%20==0:print('Trainingroots',i+1,'/120 labels',len(rows),flush=True);save('training-labels.json',rows)
    save('training-labels.json',rows)
    if len(rows)<120:raise ValueError('Too few diverse quiet search labels')
    x=torch.tensor([r['features'] for r in rows],dtype=torch.float32);y=torch.tensor([max(-400,min(400,r['teacher_white_cp']-r['base_white_cp'])) for r in rows],dtype=torch.float32)
    counts=Counter(r['category'] for r in rows);importance=torch.tensor([1/counts[r['category']] for r in rows]);importance/=importance.sum()
    torch.set_num_threads(1);torch.manual_seed(260031)
    w=torch.nn.Parameter(torch.tensor(PRIORS));prior=torch.tensor(PRIORS);bounds=torch.tensor(BOUNDS);opt=torch.optim.Adam([w],lr=.03)
    for step in range(600):
        opt.zero_grad();errors=torch.nn.functional.smooth_l1_loss((x@w)/400,y/400,reduction='none')
        loss=(errors*importance).sum()+.01*((w-prior)/bounds).square().mean();loss.backward();opt.step()
        with torch.no_grad():w.clamp_(min=0);w.copy_(torch.minimum(w,bounds))
    weights=w.detach().tolist();save('fitted-weights.json',{'names':FEATURE_NAMES,'weights':weights,'priors':PRIORS,'bounds':BOUNDS,'training_rows':len(rows),'categories':dict(counts),'objective':'Category-balanced Huber on64000-node actualquietsearchleaf residual, clipped+-400cp; ridge.01 togeneralpriors,600fixedsteps. Material/PSTfixed, noNN.'})
    return weights

def screen(sf,roots,models,modes,filename):
    records=[]
    for i,r in enumerate(roots):
        probes={};cache={}
        names=list(models);names=names[i%len(names):]+names[:i%len(names)]
        for mode in modes:
            opts={'depth':3} if mode=='depth3' else {'depth':64,'time_limit':.75}
            ps={}
            for n in names:
                model=native if models[n] is None else StrategicEvaluator(models[n])
                p=search(reconstruct(r),eval_fn=model,use_lmr=True,**opts)
                ps[n]={'move':p.move.uci(),'depth':p.depth,'nodes':p.nodes,'elapsed':p.elapsed}
            for n,p in ps.items():
                if p['move'] not in cache:cache[p['move']]=review(sf,reconstruct(r),chess.Move.from_uci(p['move']),256000)
                p['review']=cache[p['move']]
            probes[mode]=ps
        records.append({'root':r,'probes':probes});save(filename,records)
        if (i+1)%10==0:print(filename,i+1,'/',len(roots),flush=True)
    summary={}
    for mode in modes:
        summary[mode]={}
        for name in models:
            def summarize(rs):
                ps=[r['probes'][mode][name] for r in rs]
                common=[r for r in rs if all(r['probes'][mode][n]['review']['cp_loss'] is not None for n in models)]
                losses=[r['probes'][mode][name]['review']['cp_loss'] for r in common]
                return {'positions':len(ps),'comparable':len(losses),'mean_cp_regret':sum(losses)/len(losses) if losses else None,'bad_150cp_moves':sum((p['review']['cp_loss'] or 0)>=150 for p in ps),'losing_transitions':sum(fatal(p['review']) for p in ps),'allowed_mates':sum(p['review']['allows_mate'] for p in ps),'mean_depth':sum(p['depth'] for p in ps)/len(ps)}
            summary[mode][name]={**summarize(records),'categories':{cat:summarize([r for r in records if r['root']['category']==cat]) for cat in CATEGORIES}}
    save(filename.replace('.json','-summary.json'),summary)
    return summary

def qualifies(a,z,strict=True):
    if z['comparable']<32 or a['mean_cp_regret'] is None or z['mean_cp_regret'] is None:return False
    return (z['mean_cp_regret']<a['mean_cp_regret'] if strict else z['mean_cp_regret']<=a['mean_cp_regret']) and all(z[k]<=a[k] for k in ('bad_150cp_moves','losing_transitions','allowed_mates'))

def main():
    if (ART/'selection.json').exists():raise FileExistsError('Preserve modelselection')
    train=json.loads((ART/'train-roots.json').read_text());val=json.loads((ART/'val-roots.json').read_text());test=json.loads((ART/'test-roots.json').read_text())
    with exclusive_cpu('v26 profiling jointweight fitting and searched validation'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32});profiling(val)
        weights=(json.loads((ART/'fitted-weights.json').read_text())['weights']
                 if (ART/'fitted-weights.json').exists() else collect_and_fit(sf,train,val,test))
        save('position-cache-amendment.json',{'reason':'Fixeddepthprofile showed first reuse implementation18.8%slower. Strategic scores do not encodehalfmoveclock; explicit position_only allows safe reuse while terminal andhistory checks remain unchanged. Prior developmentprobes preserved, no finaltest consulted. Trainingdata/weights unchanged; rerunprofile andall developmentcontrols.'})
        models={'baseline':None,'cached_zero':[0.]*6,'prior':list(PRIORS),'fitted':weights}
        result=screen(sf,val,models,('depth3','750ms'),'validation.json')
        good=[name for name in models if name!='baseline' and all(qualifies(result[mode]['baseline'],result[mode][name],strict=(mode=='750ms')) for mode in result)]
        selected=min(good,key=lambda n:(result['750ms'][n]['mean_cp_regret'],result['depth3'][n]['mean_cp_regret'])) if good else None
        save('selection.json',{'selected':selected,'weights':models.get(selected),'development_gate_passed':bool(selected),'policy':'Fixed4configs; select amongnonregressing depth3 and strictlybetter750ms, thenlower750ms/depth3 regret. Finaltestnever selects weights. Cachedzero mustmatch originalexactdepth scores, butmaywin through exactevaluationreuse.'})
        if selected:
            final=screen(sf,test,{'baseline':None,'candidate':models[selected]},('depth3','750ms'),'test.json')
            passed=all(qualifies(final[m]['baseline'],final[m]['candidate'],strict=(m=='750ms')) for m in final)
            save('decision.json',{'qualifies_for_100_games':passed,'selected':selected,'app_promoted':False})
        else:save('decision.json',{'qualifies_for_100_games':False,'selected':None,'reason':'No config passed developmentsearched-choice gate','app_promoted':False})

if __name__=='__main__':main()
