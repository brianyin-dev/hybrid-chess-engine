"""Replay and verify the completed user-requested v35 paired match."""
import json,hashlib,io
from pathlib import Path
import chess,chess.pgn,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import opening_board
from ml.run_search_v22 import exact
from ml.run_balanced_v19 import SF

def main():
    root=Path(__file__).resolve().parents[1];out=root/'ml/artifacts/relationship-v35/user-pilot-10-250ms';r=json.load(open(out/'report.json'))
    assert r['status']=='completed' and len(r['games'])==10 and r['config']['time_ms']==250
    assert hashlib.sha256((root/'ml/artifacts/relationship-v35/current-epoch-40.pt').read_bytes()).hexdigest()==r['checkpoint_sha256']
    for p,h in r['source_sha256'].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
    fallback=[]
    for i,g in enumerate(r['games'],1):
     b=chess.Board(g['initial_fen'])
     for u in g['opening_moves']:b.push_uci(u)
     for m in g['moves']:
      mv=chess.Move.from_uci(m['uci']);assert mv in b.legal_moves
      if m['depth']==0:fallback.append({'game':i,'fen':b.fen(),**m})
      b.push(mv)
     assert b.fen()==g['final_fen'] and b.result(claim_draw=False)==g['result']
    for pair in range(1,6):
     gs=[g for g in r['games'] if g['pair']==pair]
     assert len(gs)==2 and {g['current_color'] for g in gs}=={'white','black'} and gs[0]['opening_moves']==gs[1]['opening_moves']
    s=(out/'games.pgn').read_text().rstrip()+'\n';(out/'games.pgn').write_text(s);f=io.StringIO(s);count=0
    while (g:=chess.pgn.read_game(f)) is not None:assert not g.errors;count+=1
    assert count==10
    starts=[]
    with exclusive_cpu('audit completed v35 pilot starts'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
     sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False})
     for o in r['openings']:
      b=opening_board(o);info=exact(sf,b,nodes=128000)
      starts.append({'opening':o['name'],'fen':b.fen(),'white_cp':info['score'].white().score(),'white_mate':info['score'].white().mate(),'depth':info.get('depth'),'nodes_per_search':128000})
    audit={'passed':True,'legal_replays':10,'pgn_games':count,'color_swapped_identical_start_pairs':5,'checkpoint_and_source_hashes_verified':True,'fallback_moves':fallback,'opening_reviews':starts,'policy':'Opening scores reviewed only after completed frozen match; no filtering or tuning based on them.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'summary':r['summary'],'mean_depth':r['mean_depth'],'opening_reviews':starts},indent=2))

if __name__=="__main__":main()
