import hashlib
import json
import numpy as np
import pytest
import soundfile as sf
from src.synthetic_quality_review import QualityReview
from src.counterfactual_token_evaluation import align_target
from src.validate_synthetic_quality import validate

def test_target_alignment_does_not_count_context_changes_as_target_errors():
    x=align_target('we discussed diabetes then said','metformin','today','we discussed hypertension then said metformin today')
    assert x['target_lexical_agreement']
    assert x['outcome_label'] is None
    x=align_target('take','metformin','today','take met for min today')
    assert not x['target_lexical_agreement']
    assert x['outcome_label'] is None
    x=align_target('','drug','drug','drug')
    assert x['alignment_nonunique']

def test_staging_revisions_and_audio_tampering(tmp_path):
    audio=tmp_path/'audio.wav';sf.write(audio,np.ones(1600)*.1,16000)
    digest=hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps({'audio_path':'audio.wav','audio_sha256':digest,'target_start_sample':0,'target_end_sample':800})+'\n')
    app=QualityReview(manifest,tmp_path/'events.jsonl')
    b={'case_id':'S001','reviewer_id':'reviewer-01','stage':'B','pronunciation':'yes','context_plausibility':'uncertain','splice_naturalness':'no'}
    with pytest.raises(ValueError):app.save(b)
    a=app.save({'case_id':'S001','reviewer_id':'reviewer-01','stage':'A','heard':'unclear','intelligibility':'uncertain'})
    b['stage_a_event_id']=a['event_id'];app.save(b);b['notes']='revision';app.save(b)
    assert len(app.history())==3
    assert app.history()[0]['heard']=='unclear'
    verified=validate(manifest,app.events,tmp_path/'valid.json')
    assert verified['status']=='VALID_BOUND_EVENTS' and verified['completed_cases']==1
    lines=app.history();lines[-1]['source_audio_sha256']='wrong'
    app.events.write_text(''.join(json.dumps(e)+'\n' for e in lines))
    invalid=validate(manifest,app.events,tmp_path/'invalid.json')
    assert invalid['status']=='VALIDATION_ERRORS'
    audio.write_bytes(b'changed')
    with pytest.raises(ValueError):app.save(b)
