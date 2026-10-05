"""Post-match full-strength Stockfish opening-balance audit; no selection."""
import hashlib,json
from collections import Counter
from pathlib import Path
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import opening_board

OUT=Path('ml/artifacts/standpat-v32/user-pilot-20')
SF=Path('tools/stockfish-sf19/stockfish/stockfish-macos-universal')

def main():
 with exclusive_cpu('v32 20-game pilot starting-evaluation audit'):
  match=json.loads((OUT/'report.json').read_text());engine=chess.engine.SimpleEngine.popen_uci(str(SF))
  engine.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False,'Skill Level':20})
  engine_id=dict(engine.id)
  rows=[]
  try:
   for i,o in enumerate(match['openings'],1):
    b=opening_board(o);row={'pair':i,'opening':o,'fen':b.fen(),'analyses':{}}
    for nodes in (256000,1000000):
     a=engine.analyse(b,chess.engine.Limit(nodes=nodes),game=object())
     row['analyses'][str(nodes)]={'white_cp':a['score'].white().score(),'white_mate':a['score'].white().mate(),'depth':a['depth'],'actual_nodes':a['nodes'],'pv':[m.uci() for m in a.get('pv',[])[:10]]}
    games=[g for g in match['games'] if g['pair']==i]
    row['nn_results']={g['current_color']:g['current_result'] for g in games}
    rows.append(row);print(json.dumps(row),flush=True)
  finally:engine.quit()
  cps=[r['analyses']['1000000']['white_cp'] for r in rows]
  outcomes=Counter('draw' if g['result']=='1/2-1/2' else 'white_win' if g['result']=='1-0' else 'black_win' for g in match['games'])
  summary={'min_white_cp':min(cps),'max_white_cp':max(cps),'mean_white_cp':sum(cps)/len(cps),
           'within_50_cp':sum(abs(v)<=50 for v in cps),'within_100_cp':sum(abs(v)<=100 for v in cps),
           'above_150_cp_absolute':sum(abs(v)>=150 for v in cps),'game_outcomes_by_color':dict(outcomes)}
  out={'purpose':'Post-match descriptive balance audit; openings not changed or selected using this analysis',
       'stockfish_sha256':hashlib.sha256(SF.read_bytes()).hexdigest(),'stockfish_id':engine_id,
       'options':{'Threads':1,'Hash':32,'UCI_LimitStrength':False,'Skill Level':20},'nodes_per_start':[256000,1000000],
       'summary':summary,'starts':rows}
  (OUT/'opening-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
