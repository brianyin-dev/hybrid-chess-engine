"""Prefit coverage supplement: fresh standard book prefixes in reserved families."""
import hashlib,json,random
from pathlib import Path
from collections import Counter
import chess
from engine.opening_book import choose_book_move
from benchmarks.match import opening_board
from ml.run_search_correction_v29 import family

ROOT=Path(__file__).resolve().parents[1]
def main():
 allocation=json.loads((ROOT/'ml/artifacts/standpat-v32/allocation.json').read_text())
 owners={o['family']:s for s,os in allocation.items() for o in os}
 out=ROOT/'benchmarks/openings-decision-v33-data.json'
 if out.with_suffix('.coverage-manifest.json').exists():raise FileExistsError('Preserve frozen coverage supplement')
 rows=json.loads(out.read_text());rng=random.Random(330040)
 seen={' '.join(opening_board(o).fen().split()[:4]) for p in (ROOT/'benchmarks').glob('openings*.json') for o in (lambda d:d if isinstance(d,list) else [])(json.loads(p.read_text()))}
 counts=Counter();targets={'train':90,'val':45,'test':45}
 # Existing unique starts plus a documented shorter/longer prefix supplement.
 for o in rows:
  f=family(o['moves'])
  if f in owners:counts[(owners[f],f)]+=1
 for attempt in range(100000):
  b=chess.Board();moves=[];length=rng.choice([6,8,18,20])
  for _ in range(length):
   choice=choose_book_move(b,ROOT/'books/gm2001.bin',rng=rng)
   if choice is None:break
   moves.append(choice.move.uci());b.push(choice.move)
  if len(moves)!=length:continue
  f=family(moves);s=owners.get(f);alias=' '.join(b.fen().split()[:4])
  if s is None or alias in seen or b.is_game_over() or counts[s,f]>=4:continue
  rows.append({'name':f'v33 coverage book start {len(rows)+1}','moves':moves});seen.add(alias);counts[s,f]+=1
  totals={split:sum(min(v,3) for (z,f),v in counts.items() if z==split) for split in targets}
  if all(totals[s]>=targets[s] for s in targets):break
 out.write_text(json.dumps(rows,indent=2)+'\n')
 summary={'seed':330040,'attempts':attempt+1,'total_starts':len(rows),'supplement_prefix_plies':[6,8,18,20],
          'available_counts_capped3':{s:sum(min(v,3) for (z,f),v in counts.items() if z==s) for s in targets},
          'selection':'Prefit coverage, existing v32 reserved families only, unique prior start FEN; no engine-score/model-result filtering.',
          'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 out.with_suffix('.coverage-manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':main()
