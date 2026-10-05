"""Audit actual training mix, source independence and optimized inference."""
import json
import random
import chess
import torch
from engine.strategic import PRIORS, StrategicEvaluator
from ml.generate_search_data import key
from ml.model import RELATIONAL_INPUT_SIZE, SCORE_SCALE, board_to_tensor
from ml.strategic_hybrid import StrategicHybrid
from ml.run_balanced_v19 import readrows, write, digest
from ml.run_search_correction_v29 import ART, ROOT, family
from ml import run_strategic_v28 as prior


def main():
    torch.set_num_threads(1)
    roots=json.loads((ART/'roots.json').read_text())
    mix=json.loads((ART/'training-mix.json').read_text())
    heldout={key(r['fen']) for s in ('val','test') for r in roots[s]}
    # Reproduce the exact retained sample using the frozen seed and input order.
    rng=random.Random(290029)
    oldrows=[r for r in json.loads((ART/'rebased-train.json').read_text()) if key(r['fen']) not in heldout]
    rng.shuffle(oldrows)
    raw={s:readrows(prior.DATA/f'{s}.jsonl') for s in ('train','val','test')}
    pairs={s:readrows(prior.DATA/f'pairs/{s}.jsonl') for s in raw}
    aliases={s:{key(r['fen']) for r in raw[s]} for s in raw}
    games={s:{r['game_id'] for r in raw[s]} for s in raw}
    for s in raw:
        aliases[s].update(key(r[f]) for r in pairs[s] for f in ('good_fen','bad_fen','root_fen') if f in r)
        games[s].update(r['game_id'] for r in pairs[s])
    forbidden=aliases['val']|aliases['test']
    oldpairs=[r for r in pairs['train'] if r['game_id'] not in games['val']|games['test'] and not any(key(r[f]) in forbidden|heldout for f in ('good_fen','bad_fen','root_fen') if f in r)]
    rng.shuffle(oldpairs)
    score_rows=oldrows[:2000]+mix['labels']*4
    pair_rows=oldpairs[:1500]
    train_aliases={key(r['fen']) for r in score_rows}
    train_aliases.update(key(r[f]) for r in pair_rows+mix['ranking_examples'] for f in ('good_fen','bad_fen','root_fen') if f in r)
    assert not train_aliases&heldout
    families={s:{r['family'] for r in rs} for s,rs in roots.items()}
    overlaps={f'{a}:{b}':len(families[a]&families[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(overlaps.values())
    source=json.loads((ART/'source-games.json').read_text())
    for r in source:
        b=chess.Board()
        for move in r['history']:b.push_uci(move)
        assert b.fen()==r['final_fen'] and family(r['history'])==r['family']
    for rs in roots.values():
        for r in rs:
            b=chess.Board(r['initial_fen'])
            for move in r['history']:b.push_uci(move)
            assert b.fen()==r['fen']
    inference={};base=StrategicEvaluator(PRIORS)
    positions=[r['fen'] for r in roots['val']+roots['test']]+[r['fen'] for r in mix['labels'][:68]]
    for path in sorted(ART.glob('weight-*-epoch-*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True);w=saved['training_correction_weight'];model=StrategicHybrid(path,w)
        maximum=0
        for fen in positions:
            b=chess.Board(fen)
            factor=0. if b.is_check() or next(b.generate_legal_captures(),None) else w
            with torch.inference_mode():expected=round(base(b)+factor*model.neural.model(board_to_tensor(b,RELATIONAL_INPUT_SIZE).unsqueeze(0)).item()*SCORE_SCALE)
            maximum=max(maximum,abs(model.evaluate_position(b)-expected))
            assert model.evaluate_position(b)==-model.evaluate_position(b.mirror())
            assert b.fen()==fen
        assert maximum<=1
        inference[path.name]={'positions':len(positions),'max_rounding_difference_cp':maximum,'color_symmetry':True}
    write(ART/'retention-score-rows.json',oldrows[:2000]);write(ART/'retention-pairs.json',pair_rows)
    protocol=json.loads((ART/'protocol.json').read_text())
    assert all(digest(ROOT/p)==h for p,h in protocol['source_hashes'].items())
    write(ART/'audit.json',{'source_trajectories_legal':len(source),'root_counts':{s:len(rs) for s,rs in roots.items()},
        'fresh_family_counts':{s:len(fs) for s,fs in families.items()},'fresh_family_overlaps':overlaps,
        'training_vs_heldout_root_aliases':0,'unique_new_score_rows':mix['new_unique_score_rows'],
        'training_score_rows_with_sampling':len(score_rows),'retention_pairs':len(pair_rows),
        'source_hashes_match':True,'inference':inference,'nn_promoted':False,'game_pilot_run':False,
        'limitations':'Legacy family provenance incomplete; endpoint capacity permits, not proves, root repairs.'})
    print(json.dumps(json.loads((ART/'audit.json').read_text()),indent=2))

if __name__=='__main__':main()
