"""Descriptive endpoint ranking results; never selects a checkpoint."""
import json
from collections import Counter
import chess
from engine.strategic import PRIORS,StrategicEvaluator
from ml.strategic_hybrid import StrategicHybrid
from ml.run_balanced_v19 import write
from ml.run_ranking_v30 import ART,INITIAL,WEIGHT


def main():
    data=json.loads((ART/'collection.json').read_text())
    selection=json.loads((ART/'selection.json').read_text());selected=selection['selected']
    models={'heuristic':StrategicEvaluator(PRIORS),'previous_nn':StrategicHybrid(INITIAL,WEIGHT)}
    for path in sorted(ART.glob('epoch-*.pt')):models[path.stem]=StrategicHybrid(path,WEIGHT)
    output={'selection_uses_these_metrics':False,'validation':{},'training':{}}
    for split,label in [('train','training'),('val','validation')]:
        for name,model in models.items():
            counts=Counter()
            for p in data['pairs'][split]:
                gap=p['sign']*(model(chess.Board(p['good_fen']))-model(chess.Board(p['bad_fen'])))
                counts['pairs']+=1;counts['correct']+=int(gap>0)
                counts[p['kind']+'_pairs']+=1;counts[p['kind']+'_correct']+=int(gap>0)
                counts['retention_regressions']+=int(p['kind']=='retention' and gap<=0)
            output[label][name]=dict(counts)
    # Test endpoint metrics only after the selected candidate has undergone its
    # predeclared searched test. Rejected development snapshots never see these.
    if selected and (ART/'test-summary.json').exists():
        output['test']={}
        for name in ('heuristic',selected):
            counts=Counter();model=models[name]
            for p in data['pairs']['test']:
                gap=p['sign']*(model(chess.Board(p['good_fen']))-model(chess.Board(p['bad_fen'])))
                counts['pairs']+=1;counts['correct']+=int(gap>0)
                counts[p['kind']+'_pairs']+=1;counts[p['kind']+'_correct']+=int(gap>0)
                counts['retention_regressions']+=int(p['kind']=='retention' and gap<=0)
            output['test'][name]=dict(counts)
    write(ART/'endpoint-metrics.json',output)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
