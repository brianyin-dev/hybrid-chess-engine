"""Recorded post-validation amendment; no training or test-driven retuning."""
import json
import chess
import chess.engine
import torch
from benchmarks.cpu_lock import exclusive_cpu
from engine.strategic import PRIORS,StrategicEvaluator
from ml import run_strategic_v28 as common
from ml.run_standpat_v32 import ART,ROOT,gate,pilot
from ml.run_balanced_v19 import SF,write,digest
from ml.standpat_hybrid import StandPatHybrid


def main():
    torch.set_num_threads(1)
    assert not (ART/'test-summary.json').exists(), 'Never repeat or retune on the test'
    development=json.loads((ART/'development-summary.json').read_text())
    records=json.loads((ART/'development.json').read_text())
    candidates=[]
    for name in ('epoch-8','epoch-16','epoch-24'):
        if not all(development[mode][name]['mean_regret']<development[mode]['heuristic']['mean_regret']
                   and development[mode][name]['comparable']>=20
                   and all(development[mode][name][k]<=development[mode]['heuristic'][k]
                           for k in ('losing_transitions','allowed_mates')) for mode in ('depth3','250ms')):
            continue
        extra=[]
        for mode in ('depth3','250ms'):
            for r in records:
                h=r['probes'][mode]['heuristic']['review'];n=r['probes'][mode][name]['review']
                if (n['cp_loss'] or 0)>=150 and (h['cp_loss'] or 0)<150:
                    extra.append({'root':r['root'],'mode':mode,'heuristic_review':h,'candidate_review':n})
        if all(e['candidate_review']['best_score']['cp'] is not None
               and e['candidate_review']['best_score']['cp']<=-150 for e in extra):
            candidates.append((name,extra))
    assert len(candidates)==1, 'This amendment authorizes one identified candidate, not a grid'
    name,extra=candidates[0]; checkpoint=ART/f'{name}.pt'
    protocol={'amendment_timing':'After original validation failure, before any heldout searched result. Original selection/decision/protocol are unchanged.',
              'candidate':name,'checkpoint_sha256':digest(checkpoint),
              'rationale':'Only candidate improving both fresh validation means with no extra losing transitions/mates. Its sole extra large error is already losing under teacher best cp<=-150. This exception was not predeclared and the result is exploratory.',
              'extra_errors':extra,'test':'One frozen untouched24root test, original strict gate including all large errors retained. No test tuning; pilot20 only if this independent strict test passes. No app promotion.',
              'hashes':{p:digest(ROOT/p) for p in ('ml/explore_standpat_v32.py','ml/run_standpat_v32.py','ml/standpat_hybrid.py','ml/standpat_features.py')},
              'fresh_roots_sha256':digest(ART/'fresh-roots.json')}
    path=ART/'exploratory-protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==protocol
    else:
        write(path,protocol)
    with exclusive_cpu('v32 recorded exploratory independent test and conditional pilot'):
        common.ART=ART
        roots=json.loads((ART/'fresh-roots.json').read_text())
        with chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
            sf.configure({'Threads':1,'Hash':32})
            summary=common.screen(sf,roots['test'],{'heuristic':StrategicEvaluator(PRIORS),name:StandPatHybrid(checkpoint)},'test')
        passed=gate(summary,name)
        write(ART/'exploratory-decision.json',{'candidate':name,'original_validation_gate_passed':False,
                                            'strict_independent_test_passed':passed,'qualifies_for_exploratory_pilot':passed,
                                            'app_unchanged':True})
        if passed:
            pilot(checkpoint)


if __name__=='__main__':
    main()
