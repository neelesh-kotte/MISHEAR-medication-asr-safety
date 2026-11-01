"""Validate saved quality judgments without creating ASR correctness labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from .synthetic_quality_review import QualityReview

def validate(manifest,events,output):
    if output.exists():raise ValueError('Use a new validation output')
    app=QualityReview(manifest,events);raw=events.read_bytes() if events.exists() else b''
    rows=[json.loads(x) for x in raw.splitlines() if x.strip()];errors=[];accepted={};prior={};last_time=None
    for expected,e in enumerate(rows,1):
        try:
            if e['event_id']!=expected:raise ValueError('Nonsequential event ID')
            cid=e['case_id'];r=app.rows[cid];key=(cid,e['reviewer_id'])
            time=datetime.fromisoformat(e['timestamp_utc'])
            if time.tzinfo is None or (last_time is not None and time<last_time):raise ValueError('Timestamp chronology')
            last_time=time
            if e['manifest_sha256']!=app.manifest_hash or e['source_audio_sha256']!=r['audio_sha256']:raise ValueError('Manifest/audio mismatch')
            if e['served_target_wav_sha256']!=hashlib.sha256(app.payload(cid)).hexdigest():raise ValueError('Served target mismatch')
            if e['target_samples']!=[r['target_start_sample'],r['target_end_sample']]:raise ValueError('Boundary mismatch')
            if e['stage']=='A':
                if e['intelligibility'] not in {'clear','uncertain','unintelligible'}:raise ValueError('Invalid intelligibility')
                prior[key]=e;accepted.pop(key,None)
            elif e['stage']=='B':
                if key not in prior or e['stage_a_event_id']!=prior[key]['event_id']:raise ValueError('Missing/mismatched preceding Stage A')
                fields=[e[k] for k in ('pronunciation','context_plausibility','splice_naturalness')]
                if any(x not in {'yes','no','uncertain'} for x in fields):raise ValueError('Invalid quality judgment')
                state='quality_pass_single_reviewer' if all(x=='yes' for x in fields) and prior[key]['intelligibility']=='clear' else 'quality_needs_review'
                accepted[key]={'case_id':cid,'reviewer_id':e['reviewer_id'],'status':state,'stage_a_event_id':prior[key]['event_id'],'stage_b_event_id':e['event_id'],'audio_sha256':r['audio_sha256']}
            else:raise ValueError('Unknown stage')
        except (KeyError,ValueError,TypeError) as exc:errors.append({'event_index':expected,'error':str(exc)})
    result={'status':'VALID_BOUND_EVENTS' if not errors else 'VALIDATION_ERRORS','manifest_sha256':app.manifest_hash,'events_sha256':hashlib.sha256(raw).hexdigest(),'events':len(rows),'stage_a_cases':len({k[0] for k in prior}),'completed_cases':len({k[0] for k in accepted}),'case_reviews':list(accepted.values()),'errors':errors,'limit':'Owner single-reviewer quality observations, not independently adjudicated pronunciation or ASR correctness. A new Stage A invalidates completion until a matching Stage B is saved.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','events','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=validate(a.manifest,a.events,a.output);print(json.dumps({k:r[k] for k in ('status','events','stage_a_cases','completed_cases','errors')}))
if __name__=='__main__':main()
