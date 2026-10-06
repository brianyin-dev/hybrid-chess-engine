"""Instrument two recorded depth-zero failures without changing production search."""
import cProfile,hashlib,importlib,json,pstats
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from time import perf_counter
import chess
from benchmarks.cpu_lock import exclusive_cpu
from engine.strategic import StrategicEvaluator,PRIORS
from ml.relationship_hybrid import RelationshipHybrid
S=importlib.import_module('engine.search')
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ml/artifacts/relationship-v35/user-pilot-20-250ms/fallback-profile'
CP=ROOT/'ml/artifacts/relationship-v35/current-epoch-40.pt'
Original=S._Search
class Traced(Original):
    last=None
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);Traced.last=self
        self.base_stack=None;self.active=None;self.branches=[];self.iterations=[]
        self.static_seconds=0.;self.static_calls=0;self.max_ply=0;self.check_qnodes=0
        self.fallback_seconds=0
    def fallback(self,*a,**kw):
        t=perf_counter()
        try:return super().fallback(*a,**kw)
        finally:self.fallback_seconds+=perf_counter()-t
    def static(self,b):
        t=perf_counter()
        try:return super().static(b)
        finally:self.static_seconds+=perf_counter()-t;self.static_calls+=1
    def visit(self,ply,quiescence=False):
        super().visit(ply,quiescence);self.max_ply=max(self.max_ply,ply)
    def quiescence(self,b,*a):
        self.check_qnodes+=int(b.is_check())
        return super().quiescence(b,*a)
    def negamax(self,b,depth,alpha,beta,ply):
        if ply!=0:return super().negamax(b,depth,alpha,beta,ply)
        self.base_stack=len(b.move_stack);t=perf_counter();complete=False
        try:
            v=super().negamax(b,depth,alpha,beta,ply);complete=True;return v
        finally:
            self.iterations.append({'depth':depth,'seconds':perf_counter()-t,'completed':complete,
                                    'best_partial':self.root_candidate.uci() if self.root_candidate else None})
    @contextmanager
    def pushed(self,b,m):
        isroot=self.base_stack is not None and len(b.move_stack)==self.base_stack
        t=perf_counter();n=self.nodes;q=self.qnodes;st=self.static_seconds;sc=self.static_calls;complete=False
        try:
            with super().pushed(b,m):yield
            complete=True
        finally:
            if isroot:self.branches.append({'move':m.uci(),'seconds':perf_counter()-t,'nodes':self.nodes-n,
                  'qnodes':self.qnodes-q,'static_seconds':self.static_seconds-st,'static_calls':self.static_calls-sc,
                  'completed':complete})
def model(name):return RelationshipHybrid(CP) if name=='nn' else StrategicEvaluator(PRIORS)
def result(p):
    d=asdict(p);d['move']=p.move.uci() if p.move else None;return d
def main():
 OUT.mkdir(exist_ok=False)
 source=ROOT/'ml/artifacts/relationship-v35/user-pilot-20-250ms/report.json'
 games=json.loads(source.read_text())['games']
 report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'search_sha256':hashlib.sha256((ROOT/'engine/search.py').read_bytes()).hexdigest(),
         'checkpoint_sha256':hashlib.sha256(CP.read_bytes()).hexdigest(),
         'policy':'Diagnostic only, no production edits. Three uninstrumented250ms repeats per evaluator; interleaved order. Traced250ms and unconstrained-time depth1 (100000-node safety cap) have instrumentation overhead; use normal repeats for timing outcomes. cProfile of separate depth1 run perturbs runtime. Models fresh each run; full histories; exclusive CPU lock.',
         'cases':[]}
 with exclusive_cpu('trace v35 depth-zero fallback positions'):
  for gi,ply in [(11,10),(15,20)]:
   g=games[gi-1];b=chess.Board(g['initial_fen'])
   for u in g['opening_moves']:b.push_uci(u)
   for m in g['moves'][:ply-1]:b.push_uci(m['uci'])
   recorded=g['moves'][ply-1]
   assert recorded['depth']==0
   c={'game':gi,'ply':ply,'fen':b.fen(),'history':[m.uci() for m in b.move_stack],
      'recorded':recorded,'legal_moves':len(list(b.legal_moves)),'repeats':[],'traces':{}}
   for rep in range(3):
    for name in (['nn','heuristic'] if rep%2==0 else ['heuristic','nn']):
     p=S.search(b.copy(stack=True),depth=64,time_limit=.25,eval_fn=model(name),use_lmr=True)
     c['repeats'].append({'repeat':rep,'evaluator':name,**result(p)})
   for name in ['nn','heuristic']:
    for label,opts in [('250ms',{'depth':64,'time_limit':.25}),('complete_depth1',{'depth':1,'node_limit':100000})]:
     S._Search=Traced
     try:p=S.search(b.copy(stack=True),eval_fn=model(name),use_lmr=True,**opts)
     finally:S._Search=Original
     w=Traced.last
     c['traces'][name+'_'+label]={'result':result(p),'fallback_seconds':w.fallback_seconds,
            'static_seconds':w.static_seconds,'static_calls':w.static_calls,'max_ply':w.max_ply,
            'check_qnodes':w.check_qnodes,'iterations':w.iterations,'branches':w.branches}
   profiler=cProfile.Profile();e=model('nn')
   p=profiler.runcall(S.search,b.copy(stack=True),depth=1,node_limit=100000,eval_fn=e,use_lmr=True)
   profiler.dump_stats(str(OUT/f'game-{gi}-nn-depth1.prof'))
   buf=StringIO();pstats.Stats(profiler,stream=buf).strip_dirs().sort_stats('cumulative').print_stats(40)
   (OUT/f'game-{gi}-nn-depth1-profile.txt').write_text(buf.getvalue())
   c['profiled_result']=result(p);report['cases'].append(c)
   (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
   print(json.dumps({'game':gi,'normal_repeats':c['repeats'],'depth1':{name:c['traces'][name+'_complete_depth1']['result'] for name in ['nn','heuristic']}}),flush=True)
 report['status']='completed';(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
