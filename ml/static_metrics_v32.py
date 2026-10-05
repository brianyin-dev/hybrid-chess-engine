"""Post-selection static-label metrics; never select or tune checkpoints."""
import json
from collections import Counter,defaultdict
import chess
from engine.strategic import PRIORS,StrategicEvaluator
from ml.run_standpat_v32 import ART,PREVIOUS
from ml.run_balanced_v19 import write
from ml.threat_hybrid import ThreatHybrid
from ml.standpat_hybrid import StandPatHybrid
from benchmarks.cpu_lock import exclusive_cpu


def measure(rows,pairs,models):
    result={}
    for name,model in models.items():
        predictions={}; absolute=defaultdict(list)
        for r in rows:
            score=model.evaluate_position(chess.Board(r['fen']))
            predictions[r['fen']]=score
            absolute[r['category']].append(abs(score-r['score_cp']))
        counts=Counter()
        for p in pairs:
            gap=p['sign']*(predictions[p['good_fen']]-predictions[p['bad_fen']])
            counts['pairs']+=1;counts['correct']+=int(gap>0)
            counts[p['kind']+'_pairs']+=1;counts[p['kind']+'_correct']+=int(gap>0)
            counts['retention_regressions']+=int(p['kind']=='retention' and gap<=0)
        result[name]={'score_mae_cp':{c:sum(v)/len(v) for c,v in absolute.items()},
                      'positions':{c:len(v) for c,v in absolute.items()},'rankings':dict(counts)}
    return result


def main():
    with exclusive_cpu('v32 descriptive static label metrics after selection'):
        data=json.loads((ART/'collection.json').read_text());pairs=json.loads((ART/'pairs.json').read_text())
        models={'heuristic':StrategicEvaluator(PRIORS),'previous_quiet_nn':ThreatHybrid(PREVIOUS),
                **{p.stem:StandPatHybrid(p) for p in sorted(ART.glob('epoch-*.pt'))}}
        out={'used_for_selection':False,'train':measure(data['rows']['train'],pairs['train'],models),
             'validation':measure(data['rows']['val'],pairs['val'],models)}
        selected=json.loads((ART/'selection.json').read_text())['selected']
        if not selected and (ART/'exploratory-decision.json').exists():
            selected=json.loads((ART/'exploratory-decision.json').read_text())['candidate']
            out['test_selection_kind']='Post-validation exploratory amendment; original validation gate failed'
        if selected and (ART/'test-summary.json').exists():
            out['test']=measure(data['rows']['test'],pairs['test'],{n:models[n] for n in ('heuristic',selected)})
        write(ART/'static-metrics.json',out); print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
