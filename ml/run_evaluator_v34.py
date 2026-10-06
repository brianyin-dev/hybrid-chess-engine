"""Retention-first head calibration, selected on development only, then held-out screen."""
import copy,json,random
from pathlib import Path
import chess,chess.engine,torch,numpy as np
from torch import nn
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.build_strategy_v26 import reconstruct
from ml.calibrated_hybrid import CalibratedNet,CalibratedHybrid,metadata
from ml.threat_model import board_to_threat_array
from ml.standpat_hybrid import StandPatHybrid
from ml.model import SCORE_SCALE
from ml.train import rounded_score
from ml.run_balanced_v19 import write,digest,SF
from ml.run_decision_v33 import tensors,static_metrics
from ml import run_strategic_v28 as common
from engine.strategic import StrategicEvaluator,PRIORS
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'ml/artifacts/evaluator-v34';DATA=ROOT/'ml/artifacts/decision-v33'
OLD=ROOT/'ml/artifacts/standpat-v32';INITIAL=OLD/'epoch-8.pt'

def feasible_target(ref):
    # Tanh stays inside +/-25 final cp; use +/-24 for a realizable safety margin.
    return torch.minimum(torch.full_like(ref,10),ref+48).clamp(min=1)

def fit(d):
    torch.set_num_threads(1)
    old=json.loads((OLD/'collection.json').read_text())['rows']['train'];random.Random(340034).shuffle(old)
    rows=d['rows']['train']+old[:2000]
    # All head supervision retains original family splits. Never fit on pilot losses.
    pairs=[p for p in d['pairs']['train'] if p['base_margin_cp']>-48 or p['kind']=='retention']
    scores,ranks=tensors(rows,pairs);x,b,y=scores;g,z,gb,zb,sign,importance=ranks;ref=sign*(gb-zb)
    initial=torch.load(INITIAL,map_location='cpu',weights_only=True)['state_dict']
    records=[];candidates=[]
    for regularization in (.1,1.,10.):
        torch.manual_seed(340034);model=CalibratedNet();model.load_state_dict(initial)
        nn.init.zeros_(model.net[4].weight);nn.init.zeros_(model.net[4].bias)
        frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('net.4.')}
        opt=torch.optim.AdamW(model.net[4].parameters(),lr=.003,weight_decay=.01)
        for epoch in range(1,121):
            model.train();opt.zero_grad()
            gd=.25*model(g);zd=.25*model(z)
            gap=sign*(rounded_score(gb,gd,True)-rounded_score(zb,zd,True))
            target=torch.where(ref>0,ref.clamp(max=20),feasible_target(ref))
            repair=torch.relu(target-gap)/100
            protect=torch.where(ref>0,torch.relu(target-gap)/100,torch.zeros_like(gap))
            # Ordinary labels support fitting; ranked decisions and retention dominate.
            delta=.25*model(x)*SCORE_SCALE;label=(y-b).clamp(-15,15)
            loss=(repair*importance).mean()+5*protect.mean()+regularization*(delta/25).square().mean()+.1*nn.functional.smooth_l1_loss(delta/100,label/100,beta=.1)
            loss.backward();opt.step()
            if epoch%20==0:
                assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items())
                path=ART/f'head-reg-{regularization:g}-epoch-{epoch}.pt';torch.save(metadata(copy.deepcopy(model.state_dict())),path);candidates.append(path)
                metric=static_metrics(model,d['rows']['val'],d['pairs']['val'])
                row={'path':path.name,'regularization':regularization,'epoch':epoch,'validation':metric,'train':static_metrics(model,d['rows']['train'],d['pairs']['train'])};records.append(row)
                print('Head',row,flush=True)
    # Zero is an explicit control, not a neural improvement.
    zero=CalibratedNet();zero.load_state_dict(initial);nn.init.zeros_(zero.net[4].weight);nn.init.zeros_(zero.net[4].bias)
    path=ART/'zero.pt';torch.save(metadata(zero.state_dict()),path);candidates.append(path)
    records.append({'path':path.name,'regularization':None,'epoch':0,'validation':static_metrics(zero,d['rows']['val'],d['pairs']['val'])})
    selected=min(records,key=lambda r:(r['validation']['retention_damage'],-r['validation']['correct'],r['validation']['near_bound_endpoints'],r['epoch']))
    write(ART/'training.json',{'trainable_parameters':33,'rows':len(rows),'ranking_pairs_used':len(pairs),'infeasible_repair_pairs_excluded':len(d['pairs']['train'])-len(pairs),'records':records,'selected':selected['path']})
    return ART/selected['path'],candidates

def audit(paths,d):
    result={}
    for path in paths:
        adapter=CalibratedHybrid(path);maximum=0;deltas=[]
        for r in d['rows']['train'][:40]+d['rows']['val'][:40]:
            b=reconstruct(r);before=b.fen();base=adapter.baseline(b)
            with torch.inference_mode():expected=round(base+.25*adapter.model(torch.from_numpy(board_to_threat_array(b))).item()*SCORE_SCALE)
            actual=adapter.evaluate_position(b);maximum=max(maximum,abs(actual-expected));deltas.append(actual-base)
            assert maximum<=1 and actual==-adapter.evaluate_position(b.mirror()) and b.fen()==before
            assert abs(actual-base)<=25
            if path.name=='zero.pt':assert actual==base
        result[path.name]={'positions':len(deltas),'maximum_runtime_training_difference_cp':maximum,'max_correction_cp':max(map(abs,deltas)),'symmetry_and_board_integrity':True}
    write(ART/'inference-audit.json',result)

def main():
    ART.mkdir(exist_ok=True)
    if (ART/'protocol.json').exists():raise FileExistsError('Preserve frozen experiment')
    write(ART/'protocol.json',{'hypothesis':'Reduce overfitting and near-bound disruptions by freezing learned hidden features and calibrating only33 output parameters with retention-first supervision and +/-25cp final corrections.','initial_sha256':digest(INITIAL),'data_sha256':digest(DATA/'collection.json'),'fresh_roots_sha256':digest(DATA/'fresh-roots.json'),'regularization_grid':[.1,1,10],'epochs':120,'snapshots_every':20,'selection':'Validation static retention damage first, total correct second, saturation third, earliest epoch. Include exact zero control. No pilot losses used for fitting. After selection, one development searched screen, then one untouched v33 test screen if development qualifies. No games unless both screens qualify.','target':'10cp repair margin capped by base margin+48; exclude impossible repair pairs; retain correct pairs. Exact25% correction and rounding, noncheck gate. Supporting static labels clipped15cp.','source_sha256':{p:digest(ROOT/p) for p in ['ml/calibrated_hybrid.py','ml/run_evaluator_v34.py','engine/search.py','engine/strategic.py']},'production':'unchanged heuristic'})
    with exclusive_cpu('v34 frozen evaluator calibration and searched screens'):
        d=json.loads((DATA/'collection.json').read_text());selected,paths=fit(d);audit(paths,d)
        roots=json.loads((DATA/'fresh-roots.json').read_text())
        common.ART=ART
        models={'heuristic':StrategicEvaluator(PRIORS),'old_nn':StandPatHybrid(INITIAL),'candidate':CalibratedHybrid(selected)}
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            development=common.screen(sf,roots['val'],models,'development')
            passed=selected.name!='zero.pt' and common.qualifies(development,'candidate')
            write(ART/'decision.json',{'selected':selected.name,'development_passed':passed,'test_run':False,'qualifies_for_games':False,'production':'unchanged heuristic'})
            if passed:
                test=common.screen(sf,roots['test'],models,'test');passed=common.qualifies(test,'candidate')
                write(ART/'decision.json',{'selected':selected.name,'development_passed':True,'test_run':True,'qualifies_for_games':passed,'production':'unchanged heuristic'})
        print('Finished',json.dumps(json.loads((ART/'decision.json').read_text())),flush=True)
if __name__=='__main__':main()
