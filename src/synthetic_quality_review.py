"""Local staged stimulus-quality review; no ASR predictions or human-speech cases."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import random
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import soundfile as sf
from .local_annotation_review import validate_reviewer
from .run_counterfactual_asr import sha256

PAGE = '''<!doctype html><meta charset="utf-8"><title>MISHEAR synthetic quality review</title>
<style>body{max-width:800px;margin:35px auto;font:18px system-ui}button,select,input,textarea{font:inherit;margin:8px}textarea{width:95%}audio{width:100%}</style>
<h1>Synthetic stimulus quality</h1><p>Single-reviewer observations. Stage A: listen to the isolated target and record what you hear. Stage B reveals intended text and full audio. No ASR output is shown. Save each stage; revisions append. Repeated targets may reduce blinding.</p>
<label>Reviewer ID <input id="reviewer" value="reviewer-01"></label><select id="cases" onchange="load()"></select><div id="status"></div>
<audio id="audio" controls></audio><label>Speed <select onchange="audio.playbackRate=Number(this.value)"><option>1</option><option>0.75</option><option>0.5</option><option>1.25</option></select></label>
<div id="a"><textarea id="heard" placeholder="Audible name or phonetic rendering; do not guess"></textarea><label>Intelligibility <select id="clear"><option value="">Choose</option><option>clear</option><option>uncertain</option><option>unintelligible</option></select></label><button onclick="saveA()">Save initial judgment and reveal</button></div>
<div id="b" hidden><pre id="intended" style="white-space:pre-wrap"></pre><p>Full audio:</p><audio id="full" controls></audio>
<label>Target pronunciation agrees with intended name <select id="pron"><option value="">Choose</option><option>yes</option><option>no</option><option>uncertain</option></select></label><br>
<label>Context is linguistically plausible <select id="context"><option value="">Choose</option><option>yes</option><option>no</option><option>uncertain</option></select></label><br>
<label>Splice sounds natural <select id="splice"><option value="">Choose</option><option>yes</option><option>no</option><option>uncertain</option></select></label><textarea id="notes" placeholder="Uncertainty, corrections to initial judgment, or artifacts"></textarea><button onclick="saveB()">Save quality review / append revision</button></div>
<script>
const $=id=>document.getElementById(id);let cid='',stageA=null;
async function api(path,body){let r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});let j=await r.json();if(!r.ok)throw Error(j.error);return j}
async function load(){cid=$('cases').value;stageA=null;$('b').hidden=true;$('heard').value='';$('clear').value='';$('audio').src='/audio/'+cid;for(let x of ['pron','context','splice','notes'])$(x).value='';$('status').textContent='Stage A required before reveal (including resumed cases).'}
async function saveA(){try{let j=await api('/save',{case_id:cid,reviewer_id:$('reviewer').value,stage:'A',heard:$('heard').value,intelligibility:$('clear').value});stageA=j.event_id;let b=await api('/reveal?case='+cid+'&reviewer='+encodeURIComponent($('reviewer').value));$('intended').textContent=b.intended;$('full').src='/audio/'+cid+'?full=1&reviewer='+encodeURIComponent($('reviewer').value);$('b').hidden=false;$('status').textContent='Stage A saved. Intended text is now visible.'}catch(e){$('status').textContent=e.message}}
async function saveB(){try{await api('/save',{case_id:cid,reviewer_id:$('reviewer').value,stage:'B',stage_a_event_id:stageA,pronunciation:$('pron').value,context_plausibility:$('context').value,splice_naturalness:$('splice').value,notes:$('notes').value});$('status').textContent='Saved. Choose the next case above. Revisions remain in history.'}catch(e){$('status').textContent=e.message}}
api('/cases').then(j=>{for(let c of j.cases){let o=document.createElement('option');o.value=o.textContent=c;$('cases').append(o)}load()});
</script>'''

class QualityReview:
    def __init__(self, manifest: Path, events: Path):
        self.manifest = manifest.resolve(); self.manifest_hash = sha256(manifest)
        rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
        random.Random(20261031).shuffle(rows)
        if not rows: raise ValueError('Empty manifest')
        self.rows = {f'S{i:03d}':r for i,r in enumerate(rows,1)}
        self.events = events; self.lock=threading.Lock()
        for r in rows: self.audio(r)
    def audio(self,r):
        p=(self.manifest.parent/r['audio_path']).resolve()
        if self.manifest.parent not in p.parents or sha256(p)!=r['audio_sha256']: raise ValueError('Invalid audio binding')
        return p
    def history(self):
        return [json.loads(x) for x in self.events.read_text().splitlines() if x] if self.events.exists() else []
    def stage_a(self,cid,reviewer):
        return [e for e in self.history() if e['case_id']==cid and e['reviewer_id']==reviewer and e['stage']=='A' and e['manifest_sha256']==self.manifest_hash]
    def payload(self,cid,full=False):
        r=self.rows[cid]; p=self.audio(r)
        if full: return p.read_bytes()
        wave,rate=sf.read(p,start=r['target_start_sample'],stop=r['target_end_sample'],dtype='int16')
        buf=io.BytesIO();sf.write(buf,wave,rate,format='WAV',subtype='PCM_16');return buf.getvalue()
    def save(self,data):
        cid=data['case_id'];r=self.rows[cid]; reviewer=validate_reviewer(data['reviewer_id'])
        stage=data.get('stage')
        if stage=='A':
            if data.get('intelligibility') not in {'clear','uncertain','unintelligible'}: raise ValueError('Choose intelligibility')
            if data['intelligibility']=='clear' and not str(data.get('heard','')).strip(): raise ValueError('Record audible content')
            fields={k:data.get(k,'') for k in ('heard','intelligibility')}
        elif stage=='B':
            prior=self.stage_a(cid,reviewer)
            if not prior or data.get('stage_a_event_id')!=prior[-1]['event_id']: raise ValueError('Save Stage A first')
            fields={k:data.get(k,'') for k in ('pronunciation','context_plausibility','splice_naturalness','notes','stage_a_event_id')}
            if any(fields[k] not in {'yes','no','uncertain'} for k in ('pronunciation','context_plausibility','splice_naturalness')): raise ValueError('Complete all quality choices')
        else: raise ValueError('Unknown stage')
        self.audio(r)
        with self.lock:
            hist=self.history()
            event={'event_id':len(hist)+1,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'case_id':cid,'reviewer_id':reviewer,'stage':stage,**fields,'manifest_sha256':self.manifest_hash,'source_audio_sha256':r['audio_sha256'],'served_target_wav_sha256':hashlib.sha256(self.payload(cid)).hexdigest(),'target_samples':[r['target_start_sample'],r['target_end_sample']],'status':'single-reviewer-observation-not-independent-validation'}
            self.events.parent.mkdir(parents=True,exist_ok=True)
            with self.events.open('a') as f:
                f.write(json.dumps(event,sort_keys=True)+'\n');f.flush()
            return event

def handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,value,code=200,kind='application/json'):
            blob=value if isinstance(value,bytes) else json.dumps(value).encode()
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(blob)
        def do_GET(self):
            try:
                u=urlparse(self.path);q=parse_qs(u.query)
                if u.path=='/': return self.send(PAGE.encode(),kind='text/html; charset=utf-8')
                if u.path=='/cases': return self.send({'cases':list(app.rows)})
                cid=u.path.split('/')[-1] if u.path.startswith('/audio/') else q.get('case',[''])[0]
                if u.path=='/reveal' or q.get('full')==['1']:
                    if not app.stage_a(cid,q.get('reviewer',[''])[0]): raise ValueError('Stage A required')
                if u.path.startswith('/audio/'): return self.send(app.payload(cid,q.get('full')==['1']),kind='audio/wav')
                if u.path=='/reveal':
                    r=app.rows[cid];return self.send({'intended':r['prefix_text']+' '+r['target_text']+' '+r['suffix_text']})
                self.send({'error':'not found'},404)
            except (ValueError,KeyError) as e:self.send({'error':str(e)},400)
        def do_POST(self):
            try:
                if self.path!='/save': raise ValueError('Unknown route')
                if self.headers.get('Content-Type')!='application/json': raise ValueError('JSON required')
                origin=self.headers.get('Origin')
                if origin and origin!='http://'+self.headers.get('Host',''): raise ValueError('Cross-origin request rejected')
                size=int(self.headers.get('Content-Length','0'))
                if size>65536: raise ValueError('Request too large')
                event=app.save(json.loads(self.rfile.read(size)));self.send({'event_id':event['event_id']})
            except (ValueError,KeyError) as e:self.send({'error':str(e)},400)
    return Handler

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('--port',type=int,default=8766);a=p.parse_args()
    app=QualityReview(a.manifest,a.events);server=ThreadingHTTPServer(('127.0.0.1',a.port),handler(app))
    print(f'SYNTHETIC_REVIEW_READY http://127.0.0.1:{server.server_port}/ cases={len(app.rows)} events={a.events}',flush=True);server.serve_forever()
if __name__=='__main__': main()
