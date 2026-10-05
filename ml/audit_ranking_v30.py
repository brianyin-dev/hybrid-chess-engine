"""Validate broad ranking provenance, isolation, capacity and runtime parity."""
import json
from collections import Counter,defaultdict
import chess
import torch
from engine.strategic import PRIORS,StrategicEvaluator
from ml.generate_search_data import key
from ml.model import RELATIONAL_INPUT_SIZE,SCORE_SCALE,board_to_tensor
from ml.strategic_hybrid import StrategicHybrid
from ml.run_balanced_v19 import write,digest
from ml.run_ranking_v30 import ART,ROOT,TARGETS,WEIGHT,pair_key,reserved_aliases
from ml.run_search_correction_v29 import family,quiet,interval
from benchmarks.cpu_lock import exclusive_cpu


def main():
    with exclusive_cpu('v30 final data and inference audit'):
        data=json.loads((ART/'collection.json').read_text());assert data['complete']
        sources={r['game_id']:r for r in data['sources']}
        aliases={s:set() for s in TARGETS};families={s:set() for s in TARGETS};seen=set();endpoints=set();counts={};gamecounts=Counter();sampled=defaultdict(list)
        for r in sources.values():
            b=chess.Board()
            for move in r['history']:b.push_uci(move)
            assert b.fen()==r['final_fen']
            assert family(r['history'])==r['family'];families[r['split']].add(r['family'])
        for split,ps in data['pairs'].items():
            assert len(ps)==TARGETS[split]
            kinds=Counter(p['kind'] for p in ps);stages=Counter(p['stage'] for p in ps);colors=Counter()
            assert kinds['repair']>=(30 if split=='train' else 6)
            for p in ps:
                pair=pair_key(p['good_fen'],p['bad_fen']);assert pair not in seen and pair[0]!=pair[1];seen.add(pair)
                assert not set(pair)&endpoints;endpoints.update(pair)
                source=sources[p['game_id']];assert source['split']==split and source['family']==p['family']
                prefix=source['history'][:len(source['opening']['moves'])+p['source_ply']]
                b=chess.Board(p['initial_fen'])
                for move in prefix:b.push_uci(move)
                assert b.fen()==p['root_fen'];assert p['sign']==(1 if b.turn else -1)
                colors['white' if b.turn else 'black']+=1
                assert p['root_review']['cp_loss']>=15 and p['root_teacher_nodes'] in (128000,256000)
                for label in ('good','bad'):
                    leaf=chess.Board(p['initial_fen'])
                    history=p[f'{label}_history'];assert history[:len(prefix)]==prefix
                    assert history[len(prefix)]==p[f'{label}_move']
                    for move in history:leaf.push_uci(move)
                    assert leaf.fen()==p[f'{label}_fen'] and quiet(leaf)
                capacity=interval({'fen':p['good_fen']},{'fen':p['bad_fen']},p['sign'],WEIGHT)
                assert capacity['permits_correct_ranking'] and capacity==p['capacity']
                assert p['kind']==('repair' if capacity['base_margin_cp']<=0 else 'retention')
                for f in ('good_fen','bad_fen','root_fen'):aliases[split].add(key(p[f]))
                gamecounts[p['game_id']]+=1;sampled[p['game_id']].append(p['source_ply'])
            assert colors['white'] and colors['black']
            counts[split]={'pairs':len(ps),'kinds':dict(kinds),'stages':dict(stages),'colors':dict(colors),'contributing_games':len({p['game_id'] for p in ps})}
        assert max(gamecounts.values())<=5
        for plies in sampled.values():
            slots=[(ply-16)//24 for ply in plies]
            assert len(slots)==len(set(slots)) and all(b-a>=8 for a,b in zip(sorted(plies),sorted(plies)[1:]))
        for split,rs in data['roots'].items():
            for r in rs:
                aliases[split].add(key(r['fen']))
                b=chess.Board(r['initial_fen'])
                for move in r['history']:b.push_uci(move)
                assert b.fen()==r['fen']
        overlap={f'{a}:{b}':{'positions':len(aliases[a]&aliases[b]),'families':len(families[a]&families[b])} for a,b in [('train','val'),('train','test'),('val','test')]}
        assert not any(v['positions'] or v['families'] for v in overlap.values())
        reserved=reserved_aliases()
        # Add all actual novel rows/pairs used by the warm-start checkpoint.
        prior=json.loads((ROOT/'ml/artifacts/search-correction-v29/training-mix.json').read_text())
        reserved.update(key(r['fen']) for r in prior['labels'])
        reserved.update(key(r[f]) for r in prior['ranking_examples'] for f in ('good_fen','bad_fen','root_fen'))
        heldout=aliases['val']|aliases['test']
        assert not heldout&reserved,'Warm-start training positions reached holdout'
        retained=json.loads((ART/'retained-score-rows.json').read_text())
        assert not {key(r['fen']) for r in retained}&heldout
        for split,rs in data['rows'].items():
            byfen={r['fen']:r for r in rs}
            for p in data['pairs'][split]:
                assert p['sign']*(byfen[p['good_fen']]['score_cp']-byfen[p['bad_fen']]['score_cp'])==p['cp_loss']
        protocol=json.loads((ART/'protocol.json').read_text());assert all(digest(ROOT/p)==h for p,h in protocol['source_hashes'].items())
        torch.set_num_threads(1);base=StrategicEvaluator(PRIORS);inference={}
        positions=[r['fen'] for r in data['roots']['val']+data['roots']['test']]+[r['fen'] for r in data['rows']['train'][:52]]
        for path in sorted(ART.glob('epoch-*.pt')):
            model=StrategicHybrid(path,WEIGHT);maximum=0
            for fen in positions:
                b=chess.Board(fen);factor=WEIGHT if quiet(b) else 0.
                with torch.inference_mode():expected=round(base(b)+factor*model.neural.model(board_to_tensor(b,RELATIONAL_INPUT_SIZE).unsqueeze(0)).item()*SCORE_SCALE)
                maximum=max(maximum,abs(model.evaluate_position(b)-expected))
                assert model.evaluate_position(b)==-model.evaluate_position(b.mirror()) and b.fen()==fen
            assert maximum<=1
            inference[path.name]={'positions':len(positions),'max_rounding_difference_cp':maximum,'color_symmetry':True}
        write(ART/'audit.json',{'counts':counts,'source_trajectories_legal':len(sources),'fresh_cross_split_overlap':overlap,
            'retention_and_warmstart_vs_holdout_aliases':0,'unique_endpoint_pairs':len(seen),'maximum_pairs_per_game':max(gamecounts.values()),
            'source_hashes_match':True,'inference':inference,'limitations':'Pairs within a game are correlated; whole source games/families own one split. Legacy family provenance incomplete.'})
        print(json.dumps(json.loads((ART/'audit.json').read_text()),indent=2))

if __name__=='__main__':main()
