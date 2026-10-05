"""Recorded prefit training coverage extension; heldout roots and gates fixed."""
import json,random
from collections import Counter
from ml import run_decision_v33 as run
from ml.run_balanced_v19 import write,digest

def main():
    art=run.ART
    if (art/'decision.json').exists():
        decision=json.loads((art/'decision.json').read_text())
        if decision.get('trained') or decision.get('reason')!='Predeclared dataset coverage gate failed':
            raise ValueError('Only extend an unfitted insufficient-coverage collection')
    else:
        decision={'trained':False,'reason':'Prefit coverage failed:94trainpairs below120; validation24root screen capacity22 across11families.'}
        write(art/'decision-before-training-coverage-extension.json',decision)
    if (art/'training.json').exists() or (art/'training-coverage-extension.json').exists():
        raise FileExistsError('Preserve experiment; one prefit extension only')
    assigned=json.loads((art/'allocation.json').read_text());previous=json.loads((run.OLD/'allocation.json').read_text())
    train_families={o['family'] for o in previous['train']}
    rows=json.loads((run.ROOT/'benchmarks/openings-decision-v33-data.json').read_text());random.Random(330041).shuffle(rows)
    existing={tuple(o['moves']) for os in assigned.values() for o in os};counts=Counter(o['family'] for o in assigned['train'])
    candidates=[]
    for o in rows:
        f=run.family(o['moves'])
        if f in train_families and tuple(o['moves']) not in existing:candidates.append({**o,'family':f})
    original=len(assigned['train']);original_val=len(assigned['val'])
    while len(assigned['train'])<96:
        options=[o for o in candidates if counts[o['family']]<3 and tuple(o['moves']) not in existing]
        if not options:raise ValueError('Insufficient prefit extra training starts')
        o=min(options,key=lambda o:counts[o['family']]);assigned['train'].append(o);counts[o['family']]+=1;existing.add(tuple(o['moves']))
    val_families={o['family'] for o in previous['val']}
    existing_val_families={o['family'] for o in assigned['val']}
    valgroups={}
    for o in rows:
        f=run.family(o['moves'])
        if f in val_families and f not in existing_val_families and tuple(o['moves']) not in existing:
            valgroups.setdefault(f,[]).append({**o,'family':f})
    eligible=[(f,os) for f,os in sorted(valgroups.items()) if len(os)>=2]
    if len(eligible)<3:raise ValueError('Insufficient new validation-family coverage')
    for f,os in eligible[:3]:assigned['val'].extend(os[:2])
    if (art/'decision.json').exists():(art/'decision.json').rename(art/'decision-before-training-coverage-extension.json')
    (art/'allocation.json').rename(art/'allocation-before-training-coverage-extension.json')
    write(art/'allocation.json',assigned)
    for name in ('sources.json','collection.json'):
        d=json.loads((art/name).read_text());d['complete']=False;write(art/name,d)
    protocol=json.loads((art/'protocol.json').read_text());write(art/'protocol-before-training-coverage-extension.json',protocol)
    amendment={'timing':'Before any fitting or candidate predictions on fresh roots',
               'reason':'Initial teacher-only confirmed training dataset below minimum120pairs; extend independent training trajectories and validation families to meet24root diversity; do not relax filters/gates.',
               'train_games_before':original,'train_games_after':96,'validation_games_before':original_val,'validation_games_after':len(assigned['val']),
               'validation_new_families':3,'test_sources_unchanged':True,
               'fresh_roots_sha256':digest(art/'fresh-roots.json') if (art/'fresh-roots.json').exists() else None,'utility_sha256':digest(run.ROOT/'ml/extend_decision_v33_training.py')}
    write(art/'training-coverage-extension.json',amendment)
    protocol['sources']['train']=96;protocol['sources']['val']=len(assigned['val']);protocol['prefit_training_coverage_extension']=amendment
    protocol['source_sha256']['ml/extend_decision_v33_training.py']=amendment['utility_sha256'];write(art/'protocol.json',protocol)
    run.TARGETS['train']=96;run.TARGETS['val']=len(assigned['val']);run.main()
if __name__=='__main__':main()
