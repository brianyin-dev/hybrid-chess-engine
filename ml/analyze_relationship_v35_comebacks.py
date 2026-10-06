"""Review all-game trajectories and confirm potential comebacks/advantages."""
import json,hashlib
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.analyze import score_data
from ml.run_search_v22 import exact
from ml.run_balanced_v19 import SF
from ml.diagnose_relationship_v35_losses import OUT
def main():
 target=OUT/'comeback-analysis.json'
 if target.exists():raise FileExistsError(target)
 source=OUT/'report.json';r=json.loads(source.read_text())
 old={g['game']:g for g in json.loads((OUT/'loss-diagnosis.json').read_text())['games']}
 data={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'games':[],
       'policy':'NN-turn best-continuation scores from full-strength Stockfish, full game histories. Existing loss screens reused; other games screened16k nodes. Candidate peak and earlier low/later recovery endpoints independently confirmed512k. +/-150cp modest advantage/disadvantage, +/-300cp large; not proven wins. Selection uses trajectories, so extrema may miss other events; diagnosis not strength test.'}
 with exclusive_cpu('v35 comeback analysis'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
  sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
  for gi,g in enumerate(r['games'],1):
   b=chess.Board(g['initial_fen'])
   for u in g['opening_moves']:b.push_uci(u)
   rows=[];boards={}
   reuse={m['ply']:m for m in old.get(gi,{}).get('nn_moves',[])}
   for ply,m in enumerate(g['moves'],1):
    if m['engine']=='current':
     score=reuse[ply]['review']['best_score'] if ply in reuse else score_data(exact(sf,b,nodes=16000)['score'],b.turn)
     rows.append({'ply':ply,'fullmove':b.fullmove_number,'san':m['san'],'score':score});boards[ply]=b.copy(stack=True)
    b.push_uci(m['uci'])
   finite=[x for x in rows if x['score']['cp'] is not None]
   peak=max(finite,key=lambda x:x['score']['cp']) if finite else None
   candidates=[(a,z) for a in finite if a['score']['cp']<=-150 for z in finite if z['ply']>a['ply'] and z['score']['cp']>=150]
   strongest=max(candidates,key=lambda az:az[1]['score']['cp']-az[0]['score']['cp']) if candidates else None
   selected={x['ply']:x for x in ([peak] if peak else [])+ (list(strongest) if strongest else [])}
   checks={}
   for ply in selected:
    board=boards[ply];info=exact(sf,board,nodes=512000)
    checks[str(ply)]={'score':score_data(info['score'],board.turn),'depth':info.get('depth'),'best_move':info['pv'][0].uci()}
   data['games'].append({'game':gi,'result':g['current_result'],'rows':rows,'peak_screen':peak,
                        'comeback_screen':strongest,'confirmed_positions':checks})
   target.write_text(json.dumps(data,indent=2)+'\n')
   print(json.dumps({'game':gi,'result':g['current_result'],'checks':checks}),flush=True)
 data['status']='completed';target.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':main()
