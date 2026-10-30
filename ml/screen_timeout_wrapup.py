"""Frozen checkpoint: pre-timeout versus corrected search on fresh positions."""
import json
from dataclasses import asdict
from pathlib import Path
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import opening_board
from benchmarks.pre_timeout.search import search as old_search
from engine.search import search
from ml.relationship_hybrid import RelationshipHybrid
from ml.run_balanced_v19 import SF,digest,write
from ml.run_search_v22 import exact
from ml.diagnose_critical_v23 import review
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ml/artifacts/timeout-wrapup'
CP=ROOT/'ml/artifacts/relationship-v35/current-epoch-40.pt'
def pack(p):
 d=asdict(p);d['move']=p.move.uci();return d
def main():
 OUT.mkdir(exist_ok=False)
 sources=['engine/search.py','benchmarks/pre_timeout/search.py','ml/relationship_hybrid.py','ml/screen_timeout_wrapup.py']
 data={'status':'running','checkpoint_sha256':digest(CP),'source_sha256':{p:digest(ROOT/p) for p in sources},
       'policy':'Frozen v35 current epoch40,25% correction.12 fresh book starts, teacher8k continuations, snapshots8/16plies beyond prefix; no outcome filtering. Two roots/game are related. Fixeddepth1 parity then250ms sequential rotated-order probes. Teacher128k,512k for >=100cp difference or mate disagreements. Known failures separate from fresh screen. No training.',
       'fresh':[],'known':[]}
 with exclusive_cpu('timeout fix position screen'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
  sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
  openings=json.loads((ROOT/'benchmarks/openings-timeout-wrapup-screen.json').read_text())
  roots=[]
  for i,o in enumerate(openings):
   b=opening_board(o)
   for ply in range(1,17):
    if b.is_game_over():break
    b.push(exact(sf,b,nodes=8000)['pv'][0])
    if ply in (8,16) and not b.is_game_over():roots.append((i+1,b.copy(stack=True)))
  write(OUT/'fresh-roots.json',[{'game':i,'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack],'fen':b.fen()} for i,b in roots])
  for index,(i,b) in enumerate(roots):
   item={'game':i,'fen':b.fen(),'probes':{}}
   for name,fn in ([('new',search),('old',old_search)] if index%2==0 else [('old',old_search),('new',search)]):
    p=fn(b.copy(stack=True),depth=1,eval_fn=RelationshipHybrid(CP),use_lmr=True)
    item['probes'][name]={'depth1':pack(p)}
    q=fn(b.copy(stack=True),depth=64,time_limit=.25,eval_fn=RelationshipHybrid(CP),use_lmr=True)
    item['probes'][name]['250ms']=pack(q)
   a,z=[item['probes'][n]['depth1'] for n in ['old','new']]
   assert (a['move'],a['score'],a['depth'])==(z['move'],z['score'],z['depth'])
   reviews={u:review(sf,b,chess.Move.from_uci(u),128000) for u in {item['probes'][n]['250ms']['move'] for n in ['old','new']}}
   vals=[v['cp_loss'] for v in reviews.values()]
   if any(v is None for v in vals) or max(vals)-min(vals)>=100:
    reviews={u:review(sf,b,chess.Move.from_uci(u),512000) for u in reviews}
   item['reviews']=reviews;data['fresh'].append(item);write(OUT/'screen.json',data)
   print('Screen',index+1,'/',len(roots),'new/old moves',*[item['probes'][n]['250ms']['move'] for n in ['new','old']],flush=True)
  games=json.loads((ROOT/'ml/artifacts/relationship-v35/user-pilot-20-250ms/report.json').read_text())['games']
  for gi,ply in [(11,10),(15,20)]:
   g=games[gi-1];b=chess.Board(g['initial_fen'])
   for u in g['opening_moves']:b.push_uci(u)
   for m in g['moves'][:ply-1]:b.push_uci(m['uci'])
   rows=[]
   for repeat in range(3):
    for name,fn in ([('new',search),('old',old_search)] if repeat%2==0 else [('old',old_search),('new',search)]):
     p=fn(b.copy(stack=True),depth=64,time_limit=.25,eval_fn=RelationshipHybrid(CP),use_lmr=True)
     rows.append({'repeat':repeat,'engine':name,**pack(p)})
   reviews={u:review(sf,b,chess.Move.from_uci(u),512000) for u in {r['move'] for r in rows}}
   data['known'].append({'game':gi,'ply':ply,'probes':rows,'reviews':reviews});write(OUT/'screen.json',data)
 data['status']='completed';write(OUT/'screen.json',data)
 print('Position screen completed',flush=True)
if __name__=='__main__':main()
