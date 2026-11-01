"""Post-inference target-token alignment; never used as an inference feature.

Unicode casefold, alphanumeric/apostrophe tokens, unit-cost word Levenshtein.
Ties are flagged conservatively. Fragmented medication renderings require
human interpretation; lexical differences are not clinical correctness labels.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from .run_counterfactual_asr import sha256

def tokens(text):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.casefold(), flags=re.UNICODE)

def align_target(prefix, target, suffix, hypothesis):
    before=tokens(prefix);drug=tokens(target);ref=before+drug+tokens(suffix);hyp=tokens(hypothesis)
    if not drug: raise ValueError('Empty target')
    n,m=len(ref),len(hyp);lo,hi=len(before),len(before)+len(drug)
    d=[[0]*(m+1) for _ in range(n+1)];ways=[[1]*(m+1) for _ in range(n+1)]
    for i in range(n+1):d[i][0]=i
    for j in range(m+1):d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            costs=[d[i-1][j-1]+(ref[i-1]!=hyp[j-1]),d[i-1][j]+1,d[i][j-1]+1]
            d[i][j]=min(costs);ways[i][j]=min(2,sum(w for c,w in zip(costs,[ways[i-1][j-1],ways[i-1][j],ways[i][j-1]]) if c==d[i][j]))
    i,j=n,m;ops=[]
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(ref[i-1]!=hyp[j-1]):
            ops.append(('equal' if ref[i-1]==hyp[j-1] else 'substitute',i-1,j-1));i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:ops.append(('delete',i-1,None));i-=1
        else:ops.append(('insert',i,j-1));j-=1
    ops.reverse()
    target_ops=[(op,r,h) for op,r,h in ops if lo<=r<hi or (op=='insert' and r==hi)]
    recognized=[hyp[h] for op,r,h in target_ops if h is not None]
    return {'reference_tokens':ref,'hypothesis_tokens':hyp,'target_reference_token_range':[lo,hi],
            'target_alignment_operations':target_ops,'aligned_target_rendering':' '.join(recognized),
            'target_lexical_agreement':recognized==drug,'alignment_nonunique':ways[n][m]>1,
            'full_word_edit_distance':d[n][m],
            'outcome_label':None,'limit':'Lexical alignment is a review aid, not a correctness or medication-identity judgment. Boundary insertions and all alignment ties require review.'}

def evaluate(manifest,predictions,output):
    if output.exists():raise ValueError('Use a new output path')
    rows=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    preds=[json.loads(x) for x in predictions.read_text().splitlines() if x.strip()]
    byhash={}
    for p in preds:
        key=p.get('stimulus_audio_sha256')
        if not key or key in byhash:raise ValueError('Missing or duplicate prediction audio hash')
        byhash[key]=p
    result=[]
    for r in rows:
        key=r['audio_sha256'];audio=manifest.parent/r['audio_path']
        if sha256(audio)!=key:raise ValueError('Changed stimulus audio')
        if key not in byhash:raise ValueError('Missing exact audio-bound prediction')
        p=byhash[key]
        result.append({'audio_sha256':key,'case_id':r['case_id'],'voice':r['voice'],'condition':r['condition'],
                       'requested_snr_db':r['requested_snr_db'],'requested_rt60_seconds':r['requested_rt60_seconds'],
                       'reference_status':'INTENDED_TTS_TEXT_NOT_VERIFIED_SPOKEN_REFERENCE',
                       **align_target(r['prefix_text'],r['target_text'],r['suffix_text'],p['transcript'])})
    payload={'status':'REVIEW_CANDIDATES_ONLY_NO_ACCEPTED_ERRORS','manifest_sha256':sha256(manifest),
             'predictions_sha256':sha256(predictions),'normalization':'Unicode casefold; alphanumeric/apostrophe words; unit-cost word Levenshtein; diagonal/deletion/insertion tie order with nonunique flag',
             'rows':result,'accepted_error_labels':0}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2)+'\n')
    return {'rows':len(result),'status':payload['status'],'output':str(output)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','predictions','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(evaluate(a.manifest,a.predictions,a.output)))
if __name__=='__main__':main()
