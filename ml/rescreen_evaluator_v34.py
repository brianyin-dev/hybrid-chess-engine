"""Repeat frozen development screen without concurrent unittest activity."""
import json
import chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from ml.run_evaluator_v34 import ART,DATA,INITIAL
from ml.calibrated_hybrid import CalibratedHybrid
from ml.standpat_hybrid import StandPatHybrid
from engine.strategic import StrategicEvaluator,PRIORS
from ml.run_balanced_v19 import write,SF,digest
from ml import run_strategic_v28 as common

def main():
    if (ART/'development-clean.json').exists():raise FileExistsError('Preserve clean rerun')
    selected=json.loads((ART/'training.json').read_text())['selected']
    write(ART/'clean-rescreen-protocol.json',{'reason':'First development screen overlapped unit tests; benchmark CLI test waited for CPU lock and timed out. Tests subsequently passed without benchmark. Repeat fixed selection and roots under CPU lock without concurrent tests. No retraining or selection changes. Preserve initial screen.','selected_sha256':digest(ART/selected),'roots_sha256':digest(DATA/'fresh-roots.json'),'test_remains_unopened_unless_development_qualifies':True})
    common.ART=ART
    with exclusive_cpu('v34 clean fixed-development rerun'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32})
        models={'heuristic':StrategicEvaluator(PRIORS),'old_nn':StandPatHybrid(INITIAL),'candidate':CalibratedHybrid(ART/selected)}
        roots=json.loads((DATA/'fresh-roots.json').read_text())
        development=common.screen(sf,roots['val'],models,'development-clean')
        passed=selected!='zero.pt' and common.qualifies(development,'candidate')
        decision={'selected':selected,'development_passed':passed,'test_run':False,'qualifies_for_games':False,'primary_screen':'development-clean','production':'unchanged heuristic'}
        if passed:
            test=common.screen(sf,roots['test'],models,'test');decision.update(test_run=True,qualifies_for_games=common.qualifies(test,'candidate'))
        write(ART/'decision.json',decision);print(json.dumps(decision),flush=True)
if __name__=='__main__':main()
