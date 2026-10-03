"""Dataset independence, inference symmetry, legal games and matched screens."""
import io,json,subprocess
import chess,chess.pgn
from benchmarks.build_strategy_v26 import ROOT,ART,save,reconstruct
from engine.strategic import StrategicEvaluator
from benchmarks import evaluation_baseline_v26 as frozen
from benchmarks.cpu_lock import exclusive_cpu
from ml.generate_search_data import key

def main():
    with exclusive_cpu('v26 finalverification'):
        roots={s:json.loads((ART/f'{s}-roots.json').read_text()) for s in ('train','val','test')}
        labels=json.loads((ART/'training-labels.json').read_text());selection=json.loads((ART/'selection.json').read_text())
        aliases={s:{key(r['fen']) for r in rs} for s,rs in roots.items()};aliases['train'].update(key(r['fen']) for r in labels)
        families={s:{r['family'] for r in rs} for s,rs in roots.items()};families['train'].update(r['family'] for r in labels)
        overlaps={f'{a}:{z}':{'positions':len(aliases[a]&aliases[z]),'families':len(families[a]&families[z])} for a,z in [('train','val'),('train','test'),('val','test')]}
        assert not any(x['positions'] or x['families'] for x in overlaps.values())
        model=StrategicEvaluator(selection['weights'] or [0.]*6);zero=StrategicEvaluator();parity=0
        for r in sum(roots.values(),[])+labels:
            b=reconstruct(r);before=b.fen();h=list(b.move_stack)
            assert zero(b)==frozen.evaluate_position(b)
            assert model(b)==-model(b.mirror())
            assert b.fen()==before and b.move_stack==h;parity+=1
        common={}
        for name in ('validation','test'):
            path=ART/f'{name}.json'
            if not path.exists():continue
            rs=json.loads(path.read_text());models=list(rs[0]['probes']['depth3']);common[name]={}
            for mode in ('depth3','750ms'):
                subset=[r for r in rs if all(r['probes'][mode][n]['review']['cp_loss'] is not None for n in models)]
                common[name][mode]={'comparable':len(subset),'mean_regret_same_positions':{n:sum(r['probes'][mode][n]['review']['cp_loss'] for r in subset)/len(subset) for n in models},'allowed_mates':{n:sum(r['probes'][mode][n]['review']['allows_mate'] for r in rs) for n in models},'missed_forced_mates':{n:sum(r['probes'][mode][n]['review']['missed_forced_mate'] for r in rs) for n in models}}
        save('common-position-summaries.json',common)
        original=subprocess.check_output(['git','show','f3f9659:engine/evaluation.py'])
        assert (ROOT/'benchmarks/evaluation_baseline_v26.py').read_bytes()==original
        count=0;path=ART/'confirmation/report.json'
        if path.exists():
            report=json.loads(path.read_text());stream=io.StringIO((ART/'confirmation/games.pgn').read_text())
            for r in report['games']:
                g=chess.pgn.read_game(stream);assert g and not g.errors;b=g.board();moves=[]
                for m in g.mainline_moves():assert m in b.legal_moves;b.push(m);moves.append(m.uci())
                assert moves==r['opening_moves']+[m['uci'] for m in r['moves']]
                assert b.fen()==r['final_fen'];outcome=b.outcome(claim_draw=False)
                if r['current_result'] is not None:assert outcome and outcome.result()==r['result']
                count+=1
            assert chess.pgn.read_game(stream) is None
        save('audit.json',{'cross_split_overlaps':overlaps,'zero_evaluation_parity_and_color_symmetry_positions':parity,'baseline_matches_commit':'f3f9659','legal_pgn_games':count,'training_search_labels':len(labels),'nn_training_run':False})
        print(json.dumps(json.loads((ART/'audit.json').read_text()),indent=2))

if __name__=='__main__':main()
