"""Freeze diverse, current-family separated development and test positions."""
import hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import opening_board
from ml.generate_search_data import key
from ml.diagnose_critical_v23 import SF
from engine import evaluation as hce

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'benchmarks/results/strategy-v26'
CATEGORIES=('tactics','king_pressure','ordinary_middlegame','pawn_races','endgames')

def save(name,obj):(ART/name).write_text(json.dumps(obj,indent=2)+'\n')
def reconstruct(r):
    b=chess.Board(r['initial_fen'])
    for u in r['history']:b.push_uci(u)
    return b

def category(b):
    phase=hce._phase(b)
    advanced=any((s>>3 if c else 7-(s>>3))>=5 and not (b.pawns&b.occupied_co[not c]&hce.PASSED_MASKS[c][s]) for c in chess.COLORS for s in b.pieces(chess.PAWN,c))
    if phase<=12 and advanced:return 'pawn_races'
    if phase<=10:return 'endgames'
    if b.is_check():return 'tactics'
    for m in b.generate_legal_captures():
        victim=b.piece_type_at(m.to_square) or chess.PAWN
        if hce.MATERIAL[victim]-hce.MATERIAL[b.piece_type_at(m.from_square)]>=200:return 'tactics'
    for c in chess.COLORS:
        k=b.king(c)
        if k is None:continue
        ring=chess.BB_KING_ATTACKS[k]|chess.BB_SQUARES[k]
        attackers=sum(bool(b.attacks_mask(s)&ring) for s in chess.scan_forward(b.occupied_co[not c]))
        if b.queens and attackers>=2:return 'king_pressure'
    return 'ordinary_middlegame'

def main():
    ART.mkdir(parents=True,exist_ok=True);rng=random.Random(260027)
    if (ART/'manifest.json').exists():raise FileExistsError('Preserve frozen benchmark')
    save('protocol.json',{'source':'120fresh weighted-book starts, 2000-node Stockfish selfplay, max200plies; source move budget isnotlabel quality.',
        'splits':'Canonicalfirst4ply families kept whole. Beforelabels/fitting, greedily assign descending family sizes to lowest normalized train/val/test game counts (targets72/24/24). Geometry coverage audit showed original hash assignment put only5games in test; source split recorded separately. Allgameframes andsearchleaves inherit finalfamily assignment.',
        'selection':'Geometry-only categories, both sides to move, no teacher/candidate score filtering. Canonicalroot aliases excluded fromallpriorMLlabels andprevious diagnostic/test roots. Train24/val8/test8 percategory, max2pergame/category, frozenbeforelabels or fitting. Coverage amendment before any labels/fitting: replay all plies ofsaved games to avoid White-only initialsampling and insufficientkingpressure coverage.',
        'gates':'Development selects among fixed prior/fitted/zero feature weights. Final40root test uses selected candidate only; no retuning. Require lower searched regret atdepth3 and750ms with noadditional>=150cp mistakes/avoidable -150cp transitions before100fresh paired250ms games. Noapp promotion without >50% over100completedgames.',
        'nn':'No NN fitting thisround. Actualquietsearchleaf data andevaluationbenchmark are reusable for laterNNtraining.'})
    prior=set()
    for path in (ROOT/'ml/data').glob('**/*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line)
            for f in ('fen','root_fen','good_fen','bad_fen'):
                if f in r:prior.add(key(r[f]))
    for path in (ROOT/'ml/artifacts').glob('*/**/*roots.json'):
        rows=json.loads(path.read_text())
        if isinstance(rows,list):
            for r in rows:
                if isinstance(r,dict) and 'fen' in r:prior.add(key(r['fen']))
    save('prior-exclusion-count.json',{'canonical_positions':len(prior)})
    pools=defaultdict(list);games=[]
    openings=json.loads((ROOT/'benchmarks/openings-strategy-v26-data.json').read_text())
    if (ART/'source-games.json').exists():
        games=json.loads((ART/'source-games.json').read_text())
        if len(games)!=120:raise ValueError('Source generation incomplete')
    else:
      with exclusive_cpu('v26 representative benchmark source generation'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32})
        for gi,o in enumerate(openings):
            b=opening_board(o);f=chess.Board()
            for u in o['moves'][:4]:f.push_uci(u)
            family=int(hashlib.sha256(key(f.fen()).encode()).hexdigest()[:12],16)
            split='train' if family%10<6 else 'val' if family%10<8 else 'test'
            history=list(o['moves']);token=object()
            for ply in range(200):
                if b.is_game_over():break
                if b.fullmove_number>=9:
                    alias=key(b.fen())
                    if alias not in prior:
                        cat=category(b);pools[split,cat].append({'fen':b.fen(),'initial_fen':chess.STARTING_FEN,'history':history.copy(),'family':family,'source_game':gi,'split':split,'category':cat})
                p=sf.play(b,chess.engine.Limit(nodes=2000),game=token)
                history.append(p.move.uci());b.push(p.move)
            games.append({'opening':o,'history':history,'family':family,'split':split,'result':b.result()})
            if (gi+1)%10==0:print('Sourcegames',gi+1,'/120',flush=True);save('source-games.json',games)
    save('source-games.json',games)
    family_sizes=Counter(g['family'] for g in games);assigned={};split_counts=Counter();targets={'train':72,'val':24,'test':24}
    for family,size in sorted(family_sizes.items(),key=lambda x:(-x[1],x[0])):
        split=min(targets,key=lambda s:split_counts[s]/targets[s])
        assigned[family]=split;split_counts[split]+=size
    for g in games:
        g.setdefault('source_split_initial',g['split']);g['split']=assigned[g['family']]
    save('source-games.json',games);save('family-assignment.json',{'assignments':assigned,'game_counts':dict(split_counts)})
    # Replaying the preserved sources selects both sides, without moreteacher
    # computation or consulting anycandidate/label result.
    pools=defaultdict(list)
    for gi,g in enumerate(games):
        b=chess.Board();history=[]
        for u in g['history']:
            if b.fullmove_number>=9 and len(history)>=len(g['opening']['moves']) and not b.is_game_over():
                if key(b.fen()) not in prior:
                    cat=category(b);pools[g['split'],cat].append({'fen':b.fen(),'initial_fen':chess.STARTING_FEN,'history':history.copy(),'family':g['family'],'source_game':gi,'split':g['split'],'category':cat})
            b.push_uci(u);history.append(u)
    selected={s:[] for s in ('train','val','test')};seen=set();counts={}
    for s in ('test','val','train'):
        for cat in CATEGORIES:
            rows=pools[s,cat];rng.shuffle(rows);bygame=Counter();wanted=24 if s=='train' else 8
            for r in rows:
                alias=key(r['fen'])
                if alias in seen or bygame[r['source_game']]>=2:continue
                seen.add(alias);selected[s].append(r);bygame[r['source_game']]+=1
                if sum(x['category']==cat for x in selected[s])==wanted:break
            counts[s,cat]=sum(x['category']==cat for x in selected[s])
            if counts[s,cat]<wanted:raise ValueError(f'Insufficient {s}/{cat}: {counts[s,cat]}/{wanted}; preserve source, extend geometry coverage before training')
        save(f'{s}-roots.json',selected[s])
    aliases={s:{key(r['fen']) for r in rs} for s,rs in selected.items()}
    families={s:{r['family'] for r in rs} for s,rs in selected.items()}
    for a,z in (('train','val'),('train','test'),('val','test')):assert not aliases[a]&aliases[z] and not families[a]&families[z]
    save('manifest.json',{'root_counts':{s:dict(Counter(r['category'] for r in rs)) for s,rs in selected.items()},'family_counts':{s:len(x) for s,x in families.items()},'prior_alias_overlaps':0,'cross_split_family_and_alias_overlaps':0,'seed':260027})

if __name__=='__main__':main()
