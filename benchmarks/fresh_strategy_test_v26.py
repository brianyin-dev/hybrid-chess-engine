"""Freeze unseen roots after an analytical tempo correction; no score filtering."""
import json,random
from collections import Counter,defaultdict
import chess
from benchmarks.build_strategy_v26 import ROOT,ART,save,category,CATEGORIES
from ml.generate_search_data import key
from engine.strategic import StrategicEvaluator

def main():
    if (ART/'test-roots.json').exists():raise FileExistsError('Preserve newtest')
    old=json.loads((ART/'before-tempo-fix-training-labels.json').read_text())
    model=StrategicEvaluator();forbidden=set()
    for r in old:
        b=chess.Board(r['fen']);base,x=model.analyze(b)
        r.update(base_white_cp=base,features=x);forbidden.add(key(r['fen']))
    save('training-labels.json',old)
    for name in ('train-roots.json','val-roots.json','before-tempo-fix-test-roots.json'):
        forbidden.update(key(r['fen']) for r in json.loads((ART/name).read_text()))
    raw=set()
    for path in (ROOT/'ml/data').glob('**/*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line)
            raw.update(r[f] for f in ('fen','root_fen','good_fen','bad_fen') if f in r)
    forbidden.update(key(f) for f in raw)
    pools=defaultdict(list);games=json.loads((ART/'source-games.json').read_text())
    for gi,g in enumerate(games):
        if g['split']!='test':continue
        b=chess.Board();history=[]
        for u in g['history']:
            if b.fullmove_number>=9 and len(history)>=len(g['opening']['moves']) and not b.is_game_over() and key(b.fen()) not in forbidden:
                cat=category(b);pools[cat].append({'fen':b.fen(),'initial_fen':chess.STARTING_FEN,'history':history.copy(),'family':g['family'],'source_game':gi,'split':'test','category':cat})
            b.push_uci(u);history.append(u)
    rng=random.Random(260042);selected=[];seen=set()
    for cat in CATEGORIES:
        rows=pools[cat];rng.shuffle(rows);used=set();sides=Counter()
        for r in rows:
            alias=key(r['fen']);turn=chess.Board(r['fen']).turn
            if alias in seen or r['source_game'] in used or sides[turn]>=4:continue
            selected.append(r);seen.add(alias);used.add(r['source_game']);sides[turn]+=1
            if len(used)==8:break
        if len(used)<8:raise ValueError(f'Insufficient fresh {cat} independentgame/color coverage')
    save('test-roots.json',selected)
    save('tempo-correction-protocol.json',{'reason':'Analytical property violation: opponentfirsttempo increasedpassedpawnbonus. Correct kingcatchradius usespawnsteps+opponentfirsttempo, notminus. Added independentphysics consistency test. Partialconfirmation archived asinvalidforadoption.',
        'data':'Same actualquiettrainlabels andfamilies, featurecolumns recomputed, teacherlabels unchanged. Devroots unchanged; no testselection. Frozen40newtestroots,8/category,4eachside,max1/game/category, excludesalloldtest/training/legacylabel aliases. Testfamilies/sourcegames mayoverlap earlier test; positions unseen, notglobally independentnewgames.',
        'confirmation':'Use50newbookstarts, seed260041, excludeallprior starts, freezebeforecorrectedvalidation/test. Aborted oldgames never combined withnew100.',
        'seed':260042})
    print('Frozen correctedtest',len(selected),'and reusedteacherlabels',len(old))

if __name__=='__main__':main()
