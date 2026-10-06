"""Higher-budget independent confirmation of selected diagnostic positions."""
import json
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from ml.diagnose_critical_v23 import review
from ml.run_balanced_v19 import SF
from ml.diagnose_relationship_v35_losses import OUT
def main():
 p=OUT/'loss-diagnosis.json';data=json.loads(p.read_text())
 target=OUT/'loss-confirmation.json'
 if target.exists():raise FileExistsError(target)
 result={'nodes_per_search':512000,'cases':[]}
 with exclusive_cpu('confirm v35 loss cases'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
  sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
  source=json.loads((OUT/'report.json').read_text())
  for g in data['games']:
   played=source['games'][g['game']-1]
   for c in g['critical_cases']:
    b=chess.Board(played['initial_fen'])
    for u in played['opening_moves']:b.push_uci(u)
    for m in played['moves'][:c['ply']-1]:b.push_uci(m['uci'])
    u=played['moves'][c['ply']-1]['uci']
    moves={u}|{p['move'] for probes in c['probes'].values() for p in probes.values()}
    reviews={v:review(sf,b,chess.Move.from_uci(v),512000) for v in sorted(moves)}
    result['cases'].append({'game':g['game'],'ply':c['ply'],'played':u,'reviews':reviews})
   target.write_text(json.dumps(result,indent=2)+'\n')
   print('Confirmed',g['game'],flush=True)
 result['status']='completed';target.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
