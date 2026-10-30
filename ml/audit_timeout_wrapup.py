"""Preserve user-stopped raw run, extract exactly five completed games, audit."""
import copy,hashlib,io,json
from pathlib import Path
import chess,chess.pgn
from benchmarks.match import summarize
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'ml/artifacts/timeout-wrapup'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 raw=ART/'pilot-20';out=ART/'pilot-5'
 if out.exists():raise FileExistsError(out)
 source=json.loads((raw/'report.json').read_text())
 assert source['summary']['completed']==5
 report=copy.deepcopy(source);report['games']=[g for g in source['games'] if g['current_result'] is not None]
 assert len(report['games'])==5 and source['games'][:5]==report['games']
 report['excluded_interrupted_games']=[g for g in source['games'] if g['current_result'] is None]
 assert len(report['excluded_interrupted_games'])==1
 report['status']='completed_user_requested_five';report['planned_games']=5
 report['config']['pairs']=3;report['openings']=report['openings'][:3]
 report['stop_policy']='User shortened active20-game pilot to5 before completion; first5 retained irrespective of results. Stop monitor interrupted game6 after one ply. Fragment preserved in raw run and excluded. Two complete pairs plus one unmatched White game; no significance or expansion claim. UI/API smoke checks briefly overlapped this exploratory pilot.'
 report['raw_report_sha256']=digest(raw/'report.json')
 report['source_sha256']['ml/artifacts/timeout-wrapup/original-pilot-runner.py']=report['source_sha256'].pop('ml/run_timeout_wrapup_pilot.py')
 for p,h in report['source_sha256'].items():assert digest(ROOT/p)==h,p
 assert digest(ROOT/'ml/artifacts/relationship-v35/current-epoch-40.pt')==report['checkpoint_sha256']
 report['summary']=summarize(report['games'])
 report['mean_depth']={name:sum(m['depth'] for g in report['games'] for m in g['moves'] if m['engine']==name)/sum(m['engine']==name for g in report['games'] for m in g['moves']) for name in ['current','pre_timeout']}
 for g in report['games']:
  b=chess.Board(g['initial_fen'])
  for u in g['opening_moves']:b.push_uci(u)
  for m in g['moves']:assert chess.Move.from_uci(m['uci']) in b.legal_moves;b.push_uci(m['uci'])
  assert b.fen()==g['final_fen'] and b.result(claim_draw=False)==g['result']
 for pair in [1,2]:
  gs=[g for g in report['games'] if g['pair']==pair]
  assert len(gs)==2 and {g['current_color'] for g in gs}=={'white','black'} and gs[0]['opening_moves']==gs[1]['opening_moves']
 f=io.StringIO((raw/'games.pgn').read_text());pg=[]
 while (g:=chess.pgn.read_game(f)) is not None:
  assert not g.errors;pg.append(g)
 assert len(pg)==6 and pg[-1].headers['Result']=='*'
 for p,g in zip(pg[:5],report['games']):
  b=p.end().board();assert b.fen()==g['final_fen'] and p.headers['Result']==g['result']
 out.mkdir()
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'games.pgn').write_text('\n\n'.join(str(g) for g in pg[:5])+'\n')
 (out/'audit.json').write_text(json.dumps({'passed':True,'legal_replays':5,'pgn_games':5,'complete_color_pairs':2,'unmatched_game':1,'excluded_interrupted_fragment_plies':1,'checkpoint_and_source_hashes_verified':True,'original_results_preserved':True},indent=2)+'\n')
 print(json.dumps({'summary':report['summary'],'mean_depth':report['mean_depth']},indent=2))
if __name__=='__main__':main()
