"""Post-selection development metrics; never inspect test model predictions."""
import json
from collections import Counter
import chess
from ml.run_decision_v33 import ART,OLD,INITIAL
from ml.decision_hybrid import DecisionHybrid
from ml.standpat_hybrid import StandPatHybrid
from engine.strategic import StrategicEvaluator,PRIORS
from ml.run_balanced_v19 import write

def main():
    d=json.loads((ART/'collection.json').read_text());chosen=json.loads((ART/'static-selection.json').read_text())['chosen']
    models={'heuristic':StrategicEvaluator(PRIORS),'old_v32':StandPatHybrid(INITIAL),**{p:DecisionHybrid(ART/p) for p in chosen}}
    rows=d['rows']['val'];pairs=d['pairs']['val'];out={}
    for name,m in models.items():
        scores={r['fen']:m.evaluate_position(chess.Board(r['fen'])) for r in rows};correct=repair=damage=0;sat=0;group={}
        for p in pairs:
            good=p['sign']*(scores[p['good_fen']]-scores[p['bad_fen']])>0
            correct+=good;repair+=good and p['kind']=='repair';damage+=not good and p['kind']=='retention'
            for t in p['tags']:
                g=group.setdefault(t,{'pairs':0,'correct':0});g['pairs']+=1;g['correct']+=good
        sat=sum(abs(scores[r['fen']]-r['strategic_base_cp'])>=60 for r in rows)
        out[name]={'correct':correct,'pairs':len(pairs),'repair_correct':repair,'repair_total':22,'retention_damage':damage,
                   'retention_total':15,'near_bound_endpoints':sat,'endpoints':len(rows),'overlapping_tag_metrics':group}
    screens=json.loads((ART/'development.json').read_text());depths={}
    for name in screens[0]['probes']['250ms']:
        depths[name]={'mean_depth':sum(r['probes']['250ms'][name]['depth'] for r in screens)/len(screens),
                      'mean_elapsed_ms':1000*sum(r['probes']['250ms'][name]['elapsed'] for r in screens)/len(screens)}
    extras=[]
    for r in screens:
        ps=r['probes']['250ms'];h=ps['heuristic']['review']['cp_loss']
        for name in ps:
            if name not in ('heuristic','old_v32') and (ps[name]['review']['cp_loss'] or 0)>=150 and (h or 0)<150:
                extras.append({'root_fen':r['root']['fen'],'tags':r['root']['tags'],'model':name,
                               'heuristic':ps['heuristic'],'candidate':ps[name]})
    write(ART/'development-descriptive.json',{'used_for_selection':False,'test_predictions_not_inspected':True,
                                             'validation_rankings':out,'timed_search':depths,'new_large_errors':extras})
    print(json.dumps({'validation_rankings':out,'timed_search':depths,'new_large_errors':extras},indent=2))
if __name__=='__main__':main()
