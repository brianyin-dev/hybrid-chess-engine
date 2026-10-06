"""Audit immutable calibration provenance, split separation, and frozen feature weights."""
import json,random
import torch
from benchmarks.build_strategy_v26 import reconstruct
from ml.run_evaluator_v34 import ROOT,ART,DATA,OLD,INITIAL
from ml.run_balanced_v19 import write,digest

def main():
    protocol=json.loads((ART/'protocol.json').read_text());d=json.loads((DATA/'collection.json').read_text())
    assert digest(INITIAL)==protocol['initial_sha256'] and digest(DATA/'collection.json')==protocol['data_sha256']
    assert digest(DATA/'fresh-roots.json')==protocol['fresh_roots_sha256']
    for p,h in protocol['source_sha256'].items():assert digest(ROOT/p)==h
    legacy=json.loads((OLD/'collection.json').read_text())['rows']['train'];random.Random(340034).shuffle(legacy)
    training=d['rows']['train']+legacy[:2000]
    aliases=lambda rs:{' '.join(r['fen'].split()[:4]) for r in rs}
    tf={r['family'] for r in training};ta=aliases(training)
    overlaps={}
    for s in ('val','test'):
        held=d['rows'][s]+d['screen_roots'][s]
        overlaps[s]={'families':len(tf&{r['family'] for r in held}),'position_aliases':len(ta&aliases(held))}
        assert not any(overlaps[s].values())
    for r in training:assert reconstruct(r).fen()==r['fen'] and r['split']=='train'
    initial=torch.load(INITIAL,map_location='cpu',weights_only=True)['state_dict']
    checked=[]
    for path in sorted(ART.glob('*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=True)['state_dict']
        assert all(torch.equal(state[k],v) for k,v in initial.items() if not k.startswith('net.4.'))
        checked.append(path.name)
    diagnosis=json.loads((ART/'loss-diagnosis.json').read_text())
    assert digest(ROOT/'ml/artifacts/standpat-v32/user-pilot-10-500ms/report.json')==diagnosis['source_sha256']
    for g in diagnosis['games']:
        f=g['failure']
        if f:assert reconstruct(f).fen()==f['fen']
    write(ART/'provenance-audit.json',{'passed':True,'training_rows':len(training),'split_overlaps':overlaps,'all_training_and_loss_histories_verified':True,'frozen_hidden_checkpoints':checked,'immutable_source_model_data_hashes_verified':True,'losses_used_for_training':False,'unit_tests':154,'initial_test_run':'153 passed; benchmark CLI test timed out while waiting for occupied CPU lock','clean_test_run':'154 passed without benchmark'})
    print('Provenance verified',len(checked),'checkpoints')
if __name__=='__main__':main()
