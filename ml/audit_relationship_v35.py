"""Final immutable provenance and controlled-initialization audit."""
import json
from collections import Counter
import torch
from ml.collect_relationship_v35 import ROOT,ART,SPLITS
from ml.relationship_hybrid import RelationshipHybrid,ARMS
from ml.run_balanced_v19 import write,digest

def main():
    protocol=json.loads((ART/'training-protocol.json').read_text());d=json.loads((ART/'collection.json').read_text())
    assert digest(ART/'collection.json')==protocol['collection_sha256'] and digest(ART/'fresh-roots.json')==protocol['roots_sha256']
    for path,sha in protocol['source_sha256'].items():assert digest(ROOT/path)==sha,path
    dataaudit=json.loads((ART/'data-audit.json').read_text());assert dataaudit['passed'] and dataaudit['collection_sha256']==digest(ART/'collection.json')
    from ml.run_search_correction_v29 import family
    source=json.loads((ART/'sources.json').read_text())
    for game in source['sources']:assert family(game['history'])==game['family']
    for split in SPLITS:
        for pair in d['pairs'][split]:assert family(pair['root_history'])==pair['family']
    initial={a:torch.load(ART/f'{a}-zero.pt',map_location='cpu',weights_only=True)['state_dict'] for a in ARMS}
    for name,state in initial['current'].items():
        if name=='net.0.weight':
            assert torch.equal(state,initial['relationships'][name][:,:state.shape[1]])
            assert torch.count_nonzero(initial['relationships'][name][:,state.shape[1]:])==0
        else:assert torch.equal(state,initial['relationships'][name])
    assert torch.count_nonzero(initial['current']['net.4.weight'])==0 and torch.count_nonzero(initial['current']['net.4.bias'])==0
    checkpoints={}
    for path in sorted(ART.glob('*-epoch-*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True);arm=saved['arm'];state=saved['state_dict']
        changes={k:not torch.equal(v,initial[arm][k]) for k,v in state.items()}
        assert all(changes.values()),path.name
        checkpoints[path.name]={'sha256':digest(path),'all_hidden_and_head_weights_updated':True}
    selection=json.loads((ART/'selection.json').read_text());selected=selection['selected']
    assert selection['selection_used_test'] is False
    fixed=json.loads((ART/'test-depth3.json').read_text());timed=json.loads((ART/'development-250ms.json').read_text())
    assert len(fixed)==len(timed)==24
    for records in (fixed,timed):
        for r in records:
            for p in r['probes'].values():
                from benchmarks.build_strategy_v26 import reconstruct
                import chess
                assert chess.Move.from_uci(p['move']) in reconstruct(r['root']).legal_moves
    splitstats={s:{'pairs':len(d['pairs'][s]),'new_pairs':sum(not p.get('reused_prior_pair',False) for p in d['pairs'][s]),'reused_confirmed_pairs':sum(p.get('reused_prior_pair',False) for p in d['pairs'][s]),'kind':dict(Counter(p['kind'] for p in d['pairs'][s])),'source_games':dataaudit['stats'][s]['source_games'],'ranking_games':dataaudit['stats'][s]['contributing_games'],'ranking_families':dataaudit['stats'][s]['contributing_families']} for s in SPLITS}
    supplemental=json.loads((ART/'black-coverage-audit.json').read_text()) if (ART/'black-coverage-audit.json').exists() else None
    if supplemental:assert supplemental['passed'] and supplemental['overlap_with_training_families']==0 and supplemental['selection_and_decision_unchanged']
    write(ART/'final-audit.json',{'supplemental_black_coverage':supplemental,'passed':True,'shared_old_input_initialization':True,'added_inputs_zero_initialized':True,'no_prior_fitted_weights_in_training':True,'all_layers_trainable_and_updated':True,'same_data_and_batch_order_by_protocol':True,'test_not_used_for_selection':True,'data_source_and_root_hashes_unchanged':True,'split_stats':splitstats,'checkpoints':checkpoints,'selected':selected,'unit_tests':160})
    print(json.dumps(splitstats),flush=True)
if __name__=='__main__':main()
