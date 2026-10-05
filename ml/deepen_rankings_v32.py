import json
from pathlib import Path
from collections import Counter
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from ml.standpat_hybrid import StandPatHybrid
def main():
    p=Path('ml/artifacts/standpat-v32/ranking-failure-diagnosis.json');d=json.loads(p.read_text())
    rows={r['fen']:r for r in json.load(open('ml/artifacts/standpat-v32/collection.json'))['rows']['test']}
    def board(fen):
     r=rows[fen];b=chess.Board(r['initial_fen'])
     for m in r['history']:b.push_uci(m)
     return b
    def value(a):
     if a['white_cp'] is not None:return a['white_cp']
     return (100000-abs(a['white_mate']))*(1 if a['white_mate']>0 else -1)
    with exclusive_cpu('v32 diagnostic deeper confirmation'):
     e=chess.engine.SimpleEngine.popen_uci('tools/stockfish-sf19/stockfish/stockfish-macos-universal');e.configure({'Threads':1,'Hash':32})
     try:
      for i,r in enumerate(d['failures']):
       for k in ['good','bad']:
        b=board(r[k+'_fen']);a=e.analyse(b,chess.engine.Limit(nodes=256000),game=object())
        r[k+'_search_256k']={'white_cp':a['score'].white().score(),'white_mate':a['score'].white().mate(),'depth':a['depth'],'pv':[m.uci() for m in a.get('pv',[])[:12]]}
       gap=r['sign']*(value(r['good_search_256k'])-value(r['bad_search_256k']))
       r['searched_order_256k']='agrees_static' if gap>0 else 'reverses_static' if gap<0 else 'tie'
       if (i+1)%30==0:print(i+1,flush=True)
     finally:e.quit()
     c=Counter(r['searched_order_256k'] for r in d['failures']);d['deeper_counts']=dict(c)
     model=StandPatHybrid('ml/artifacts/standpat-v32/epoch-8.pt')
     for r in d['failures']:
      r['endpoint_corrections_cp']={k:model.evaluate_position(board(r[k+'_fen']))-rows[r[k+'_fen']]['strategic_base_cp'] for k in ['good','bad']}
     d['correction_near_bound_pairs']=sum(any(abs(x)>=60 for x in r['endpoint_corrections_cp'].values()) for r in d['failures'])
     p.write_text(json.dumps(d,indent=2)+'\n');print(d['deeper_counts']);print('near_bound',d['correction_near_bound_pairs'])
     for r in d['failures']:
      if r['cp_loss']>=240 and r['searched_order_256k']=='agrees_static':print(json.dumps(r))

if __name__=='__main__':main()
