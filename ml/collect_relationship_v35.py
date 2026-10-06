"""Fresh settled decision pairs from broad independent source games."""
import copy,json,random
from collections import Counter
from pathlib import Path
import chess,chess.engine
from benchmarks.cpu_lock import exclusive_cpu
from benchmarks.match import opening_board
from benchmarks.build_strategy_v26 import reconstruct
from engine.opening_book import choose_book_move
from engine.search import search,INF,_SearchLimit
from engine.strategic import StrategicEvaluator,PRIORS
from ml.run_decision_v33 import forbidden,tags
from ml.run_search_correction_v29 import family
from ml.run_search_v22 import exact
from ml.diagnose_critical_v23 import Trace
from ml.static_teacher import static_evaluation
from ml.generate_search_data import key
from ml.run_balanced_v19 import write,digest,SF
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'ml/artifacts/relationship-v35';OLD=ROOT/'ml/artifacts/decision-v33'
SPLITS=('train','val','test');EXTRA={'train':60,'val':20,'test':20}

def freeze():
    path=ART/'allocation.json'
    if path.exists():return json.loads(path.read_text())
    previous=json.loads((OLD/'sources.json').read_text());owners={g['family']:g['split'] for g in previous['sources']}
    seen={key(opening_board(g['opening']).fen()) for g in previous['sources']};rng=random.Random(350035)
    assigned={s:[] for s in SPLITS}
    for attempt in range(100000):
        b=chess.Board();moves=[];length=rng.choice([8,10,12,14,16])
        for _ in range(length):
            choice=choose_book_move(b,ROOT/'books/gm2001.bin',rng=rng)
            if choice is None:break
            moves.append(choice.move.uci());b.push(choice.move)
        if len(moves)!=length or b.is_game_over():continue
        f=family(moves);alias=key(b.fen())
        if f in owners or alias in seen:continue
        s=min(SPLITS,key=lambda z:len(assigned[z])/EXTRA[z]);owners[f]=s;seen.add(alias)
        assigned[s].append({'name':f'v35 independent {s} start {len(assigned[s])+1}','moves':moves,'family':f})
        if all(len(assigned[s])==EXTRA[s] for s in SPLITS):break
    if any(len(assigned[s])!=EXTRA[s] for s in SPLITS):raise ValueError('Insufficient independent standard-book families')
    write(path,assigned);write(ART/'allocation-manifest.json',{'seed':350035,'attempts':attempt+1,'extra_games':EXTRA,'book_sha256':digest(ROOT/'books/gm2001.bin'),'policy':'New6ply canonical families absent from all v33 source games; weighted standard book prefixes8/10/12/14/16. Inherit v33 whole-game family splits for reused histories. No score/result filtering.'})
    return assigned

def sources(sf):
    path=ART/'sources.json'
    if path.exists():data=json.loads(path.read_text())
    else:
        data={'sources':copy.deepcopy(json.loads((OLD/'sources.json').read_text())['sources']),'complete':False}
        for g in data['sources']:g['reused_history']=True
    if data['complete']:return data
    base=StrategicEvaluator(PRIORS)
    for s,os in freeze().items():
        for i,o in enumerate(os):
            gid=3500000+10000*SPLITS.index(s)+i
            if any(g['game_id']==gid for g in data['sources']):continue
            b=opening_board(o)
            for ply in range(144):
                if b.is_game_over():break
                ours=i%3==1 and b.turn==(i%2==0)
                move=search(b,depth=64,node_limit=750,eval_fn=base,use_lmr=True).move if ours else exact(sf,b,nodes=2000)['pv'][0]
                b.push(move)
            data['sources'].append({'game_id':gid,'family':o['family'],'split':s,'opening':o,'history':[m.uci() for m in b.move_stack],'final_fen':b.fen(),'truncated':not b.is_game_over(),'policy':'heuristic_vs_stockfish' if i%3==1 else 'stockfish_selfplay','reused_history':False})
            write(path,data);print('New source',s,i+1,'/',len(os),flush=True)
    data['complete']=True;write(path,data);return data

class SampleTrace(Trace):
    def __init__(self,b,base,seed):
        super().__init__(b,base,None,False,node_limit=12000);self.rng=random.Random(seed);self.sample=[];self.calls=0
    def static(self,b):
        score=super().static(b);self.calls+=1
        r={'fen':b.fen(),'initial_fen':b.root().fen(),'history':[m.uci() for m in b.move_stack]}
        index=self.rng.randrange(self.calls)
        if len(self.sample)<4:self.sample.append(r)
        elif index<4:self.sample[index]=r
        return score

def trace(root,move,base,depth):
    b=reconstruct(root);w=SampleTrace(b,base,root['game_id']+len(root['history'])+move.from_square)
    try:
        with w.pushed(b,move):score,r=w.ntrace(b,depth-1,-INF,INF,1)
    except _SearchLimit:return None,[]
    if r is None:return None,w.sample
    leaf=chess.Board(r['fen']);white=-score*(1 if b.turn else -1)
    if leaf.is_check() or leaf.is_game_over() or white!=base.evaluate_position(leaf):return None,w.sample
    row={**r,'initial_fen':root['initial_fen'],'game_id':root['game_id'],'family':root['family'],'split':root['split'],'strategic_base_cp':white,'root_fen':root['fen'],'root_move':move.uci(),'trace_depth':depth,'trace_nodes':w.nodes,'trace_qnodes':w.qnodes,'tags':tags(leaf)}
    return row,w.sample

def collect(sf,source):
    path=ART/'collection.json'
    d=json.loads(path.read_text()) if path.exists() else {'rows':{s:[] for s in SPLITS},'pairs':{s:[] for s in SPLITS},'ordinary':{s:[] for s in SPLITS},'screen_roots':{s:[] for s in SPLITS},'processed':[],'rejects':{},'complete':False}
    if d['complete']:return d
    forbidden_aliases=forbidden()
    for r in json.loads((OLD/'collection.json').read_text())['rows'].values():forbidden_aliases.update(key(x['fen']) for x in r)
    for rs in json.loads((OLD/'fresh-roots.json').read_text()).values():forbidden_aliases.update(key(r['fen']) for r in rs)
    used=forbidden_aliases|{key(r['fen']) for s in SPLITS for r in d['rows'][s]+d['ordinary'][s]+d['screen_roots'][s]}
    base=StrategicEvaluator(PRIORS);rejects=Counter(d['rejects']);processed=set(d['processed'])
    roots=[]
    for game in source['sources']:
        prefix=len(game['opening']['moves']);hist=game['history']
        for offset in (20,44,68,92,116,140):
            end=prefix+offset
            if end>=len(hist):continue
            r={k:game[k] for k in ('game_id','family','split')};r.update(initial_fen=chess.STARTING_FEN,history=hist[:end],source_ply=offset)
            b=reconstruct(r);r['fen']=b.fen()
            if not b.is_game_over() and key(b.fen()) not in forbidden_aliases:roots.append(r)
    roots.sort(key=lambda r:(SPLITS.index(r['split']),r['source_ply'],r['game_id']))
    for i,r in enumerate(roots):
        rid=f"{r['split']}:{r['game_id']}:{r['source_ply']}"
        if rid in processed:continue
        b=reconstruct(r);s=r['split'];sign=1 if b.turn else -1
        teacher=exact(sf,b,nodes=32000);cp=teacher['score'].white().score()
        if cp is None or abs(cp)>1000:rejects['root_mate_or_extreme']+=1
        else:
            if abs(cp)<=800 and key(r['fen']) not in used:
                d['screen_roots'][s].append({**r,'teacher_root_white_cp':cp,'screen_stage':'endgame' if len(b.piece_map())<=12 else 'middlegame'});used.add(key(r['fen']))
            depth=2 if r['source_ply']%48==20 else 3
            h=search(b.copy(stack=True),depth=depth,node_limit=12000,eval_fn=base,use_lmr=True)
            good=teacher['pv'][0];hscore=exact(sf,b,[h.move],nodes=64000)['score'].pov(b.turn).score()
            goodscore=exact(sf,b,[good],nodes=64000)['score'].pov(b.turn).score()
            mistake=goodscore is not None and hscore is not None and goodscore-hscore>=25
            alternatives=[h.move] if mistake and h.move!=good else []
            if not mistake:
                infos=sf.analyse(b,chess.engine.Limit(nodes=16000),multipv=min(3,b.legal_moves.count()),game=object())
                alternatives=[a['pv'][0] for a in infos if a.get('pv') and a['pv'][0]!=good][:2]
            g,ordinary=trace(r,good,base,depth)
            if s=='train':
                for sample in ordinary:
                    alias=key(sample['fen']);lb=reconstruct(sample)
                    if alias in used or lb.is_check() or lb.is_game_over():continue
                    label=static_evaluation(sf,lb)
                    if abs(label['score_cp'])>1200:continue
                    d['ordinary'][s].append({**sample,**label,**{k:r[k] for k in ('game_id','family','split')},'strategic_base_cp':base.evaluate_position(lb)});used.add(alias)
            if g is None:rejects['good_trace_limit_or_terminal']+=1
            else:
                for move in alternatives:
                    z,_=trace(r,move,base,depth)
                    if z is None:rejects['bad_trace_limit_or_terminal']+=1;continue
                    ga,za=key(g['fen']),key(z['fen'])
                    if ga==za or ga in used or za in used:rejects['duplicate_endpoint']+=1;continue
                    margin=sign*(g['strategic_base_cp']-z['strategic_base_cp'])
                    if margin<=-120 or margin>100:rejects['infeasible_or_easy']+=1;continue
                    if mistake and margin>0 or not mistake and margin<=0:rejects['trace_kind_mismatch']+=1;continue
                    ends=[]
                    for endpoint in (g,z):
                        eb=reconstruct(endpoint);a=exact(sf,eb,nodes=32000);m=a['pv'][0]
                        ends.append({'white_cp':a['score'].white().score(),'best_move':m.uci(),'depth':a.get('depth'),'quiet':not eb.is_capture(m) and not m.promotion and not eb.gives_check(m)})
                    if any(e['white_cp'] is None or not e['quiet'] for e in ends) or sign*(ends[0]['white_cp']-ends[1]['white_cp'])<15:rejects['unsettled_or_reversed_leaf']+=1;continue
                    confirmations=[]
                    for m in (good,move):
                        a=exact(sf,b,[m],nodes=128000);confirmations.append({'move':m.uci(),'mover_cp':a['score'].pov(b.turn).score(),'depth':a.get('depth')})
                    if any(a['mover_cp'] is None for a in confirmations) or confirmations[0]['mover_cp']-confirmations[1]['mover_cp']<25:rejects['root_confirmation_reverses']+=1;continue
                    g.update(static_evaluation(sf,reconstruct(g)));z.update(static_evaluation(sf,reconstruct(z)))
                    p={'good_fen':g['fen'],'bad_fen':z['fen'],'root_fen':r['fen'],'root_history':r['history'],'sign':sign,'base_margin_cp':margin,'kind':'repair' if mistake else 'retention','game_id':r['game_id'],'family':r['family'],'trace_depth':depth,'heuristic_move':h.move.uci(),'heuristic_depth_completed':h.depth,'heuristic_root_regret_screen_cp':None if hscore is None or goodscore is None else max(0,goodscore-hscore),'root_confirmation':confirmations,'endpoint_search':ends,'cp_loss':confirmations[0]['mover_cp']-confirmations[1]['mover_cp'],'tags':sorted(set(tags(b)+g['tags']+z['tags'])),'label_kind':'confirmed_search_decision_qleaf_rank_v35'}
                    d['pairs'][s].append(p);d['rows'][s]+=[g,z];used.update((ga,za));break
        d['processed'].append(rid);processed.add(rid);d['rejects']=dict(rejects)
        if len(d['processed'])%10==0:
            write(path,d);print('Roots',i+1,'/',len(roots),'pairs',{s:len(d['pairs'][s]) for s in SPLITS},'ordinary',len(d['ordinary']['train']),flush=True)
    d['complete']=True;write(path,d);return d

def freeze_screens(d):
    path=ART/'fresh-roots.json'
    if path.exists():return json.loads(path.read_text())
    chosen={}
    for s in ('val','test'):
        pool=copy.deepcopy(d['screen_roots'][s]);random.Random(350036+(s=='test')).shuffle(pool)
        used_games=set();families=Counter();stages=Counter();selected=[]
        while len(selected)<24:
            options=[r for r in pool if r['game_id'] not in used_games and families[r['family']]<2]
            if not options:raise ValueError('Insufficient independent searched roots')
            r=min(options,key=lambda r:(families[r['family']],stages[r['screen_stage']]))
            selected.append(r);used_games.add(r['game_id']);families[r['family']]+=1;stages[r['screen_stage']]+=1
        chosen[s]=selected
    write(path,chosen);return chosen

def main():
    ART.mkdir(exist_ok=True)
    if (ART/'collection-protocol.json').exists():protocol=json.loads((ART/'collection-protocol.json').read_text());assert protocol['collector_sha256']==digest(ROOT/'ml/collect_relationship_v35.py')
    else:write(ART/'collection-protocol.json',{'collector_sha256':digest(ROOT/'ml/collect_relationship_v35.py'),'reuse_source_sha256':digest(OLD/'sources.json'),'extra_games':EXTRA,'sampling':'Existing independent v33 trajectories plus100 fresh families; new roots20/44/68/92/116/140plies pastbook. One pair per root; no prior endpoints or roots; train ordinary actual static calls.','decision':'Actual heuristic depth2/3 choices screened64k; consequential >=25cp mistakes plus correct decisions; full-window traced depth2/3 +quiescence; root confirmation128k, endpoints32k quiet reply and >=15cp ranking; no requirement static SF scores agree with searched preference.','split':'Whole games and6ply canonical families; from-scratch models will not inherit prior trained weights.'})
    with exclusive_cpu('v35 diverse decision data'),chess.engine.SimpleEngine.popen_uci(str(SF)) as sf:
        sf.configure({'Threads':1,'Hash':32,'UCI_LimitStrength':False});src=sources(sf);d=collect(sf,src);freeze_screens(d)
        write(ART/'data-summary.json',{s:{'pairs':len(d['pairs'][s]),'repair':sum(p['kind']=='repair' for p in d['pairs'][s]),'contributing_games':len({p['game_id'] for p in d['pairs'][s]}),'contributing_families':len({p['family'] for p in d['pairs'][s]}),'ordinary':len(d['ordinary'][s])} for s in SPLITS})
        print('Collection complete',json.dumps(json.loads((ART/'data-summary.json').read_text())),flush=True)
if __name__=='__main__':main()
