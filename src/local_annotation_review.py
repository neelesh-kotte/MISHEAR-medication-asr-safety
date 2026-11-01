"""Loopback-only, staged human review for existing local ASR intervals.

The service never sends corpus material off-device. It streams a bounded crop
from the configured local WAV and appends annotation events without rewriting
prior responses. This is an annotation aid, not an adjudicator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audio_payload(case: Mapping[str, Any], context: bool) -> bytes:
    """Serialize the exact source-channel crop served to the reviewer."""
    import io
    import soundfile as sf

    audio_path = Path(str(case["audio_path"]))
    info = sf.info(str(audio_path))
    padding = 3.0 if context else 0.0
    begin = max(0, round(max(0.0, float(case["start_s"]) - padding) * info.samplerate))
    finish = min(info.frames, round(min(info.duration, float(case["end_s"]) + padding) * info.samplerate))
    data, rate = sf.read(str(audio_path), start=begin, frames=finish - begin,
                         dtype="float32", always_2d=True)
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _event_asset_binding(case: Mapping[str, Any], *, audio_context_used: bool) -> dict[str, Any]:
    """Revalidate and bind each saved event to source, interval, reference, and ASR assets."""
    audio_path = Path(str(case["audio_path"]))
    source_audio_hash = _sha256(audio_path)
    if source_audio_hash != case.get("audio_sha256"):
        raise ValueError("Source audio changed after the review cases were loaded; refusing to save annotation")
    provenance = dict(case.get("provenance") or {})
    textgrid_hash = provenance.get("source_textgrid_sha256")
    textgrid_value = provenance.get("source_textgrid")
    if textgrid_value:
        textgrid_path = Path(str(textgrid_value))
        if not textgrid_path.is_absolute():
            textgrid_path = audio_path.parent.parent / textgrid_path
        if textgrid_path.is_file():
            actual_textgrid_hash = _sha256(textgrid_path)
            if textgrid_hash and actual_textgrid_hash != textgrid_hash:
                raise ValueError("Source TextGrid changed after the review cases were loaded; refusing to save annotation")
            textgrid_hash = actual_textgrid_hash
    config = provenance.get("model_configuration") or {}
    if not isinstance(config, Mapping):
        config = {}
    raw_path_value = provenance.get("raw_asr_output_path")
    raw_asr_hash = None
    if raw_path_value:
        raw_path = Path(str(raw_path_value))
        if not raw_path.is_absolute():
            raw_path = Path.cwd() / raw_path
        if raw_path.is_file():
            raw_asr_hash = _sha256(raw_path)
    reviewed_audio = _audio_payload(case, audio_context_used)
    return {
        "record_id": case["record_id"],
        "source_audio_sha256": source_audio_hash,
        "source_interval_seconds": [float(case["start_s"]), float(case["end_s"])],
        "reviewed_audio_crop_sha256": hashlib.sha256(reviewed_audio).hexdigest(),
        "audio_context_included": bool(audio_context_used),
        "source_textgrid_sha256": textgrid_hash,
        "source_textgrid_reference_sha256": hashlib.sha256(str(case.get("reference", "")).encode("utf-8")).hexdigest(),
        "asr_input_crop_sha256": provenance.get("asr_input_crop_sha256"),
        "asr_hypothesis_sha256": hashlib.sha256(str(case.get("hypothesis", "")).encode("utf-8")).hexdigest(),
        "asr_raw_output_sha256": raw_asr_hash,
        "asr_run_id": provenance.get("asr_run_id"),
        "asr_prediction_artifact_sha256": provenance.get("asr_prediction_artifact_sha256"),
        "asr_run_summary_sha256": provenance.get("asr_run_summary_sha256"),
        "asr_model_configuration": {key: config.get(key) for key in
                                     ("backend", "package_version", "model_sha256", "device", "language",
                                      "temperature", "fp16", "beam_size", "best_of") if key in config},
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load a metformin case JSON or development prediction JSONL into neutral cases."""
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = value.get("cases", []) if isinstance(value, dict) else value
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Review input must contain at least one case")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_rows, 1):
        interval = row.get("audio_interval_seconds") or [row.get("interval_start_s", row.get("start_s")),
                                                           row.get("interval_end_s", row.get("end_s"))]
        audio_path = str(row.get("source_audio_path") or row.get("audio_path") or "")
        record_id = str(row.get("record_id") or row.get("utterance_id") or "")
        start, end = float(interval[0]), float(interval[1])
        audio = Path(audio_path)
        if not record_id or not audio.is_file() or start < 0 or end <= start:
            raise ValueError("Each review row needs an ID, readable local audio, and a valid positive interval")
        if row.get("source_audio_sha256") and _sha256(audio) != row["source_audio_sha256"]:
            raise ValueError(f"Audio hash changed since the case artifact was created: {audio}")
        case_id = "C%03d" % index
        if record_id in seen:
            raise ValueError(f"Duplicate record ID in review input: {record_id}")
        seen.add(record_id)
        asr = row.get("asr") or {}
        reference = row.get("source_reference_text_unreviewed", row.get("reference"))
        if reference is None:
            reference = row.get("source_textgrid_reference_posthoc_only", row.get("reference_text", ""))
        hypothesis = row.get("asr_output", row.get("hypothesis"))
        if hypothesis is None:
            hypothesis = asr.get("transcript", row.get("prediction", ""))
        provenance = dict(row.get("provenance") or {})
        provenance.update({
            "source_textgrid": row.get("source_textgrid") or row.get("source_textgrid_path") or provenance.get("source_textgrid"),
            "source_textgrid_sha256": row.get("source_textgrid_sha256") or provenance.get("source_textgrid_sha256"),
        })
        provenance.update({key: row.get(key) for key in (
            "consultation_id", "channel", "source_revision_tracked_files_only", "asr_input_crop_sha256",
        ) if row.get(key) is not None})
        provenance.update({
            "model_configuration": (row.get("model_configuration") or asr.get("model_configuration")
                                    or asr or provenance.get("model_configuration") or {}),
            "raw_asr_output_path": (row.get("raw_asr_output_path") or asr.get("raw_output_path")
                                    or provenance.get("raw_asr_output_path")),
            "raw_asr_output_sha256": (row.get("raw_asr_output_sha256") or asr.get("raw_output_sha256")
                                      or provenance.get("raw_asr_output_sha256")),
            "source_annotation_status": row.get("source_annotation_status") or row.get("source_annotation_status_posthoc_only"),
            "candidate_terms_posthoc_only": row.get("reference_selected_candidate_terms_posthoc_only") or [],
            "candidate_ingredient_ids_posthoc_only": row.get("selection_posthoc_ingredient_ids", []),
        })
        cases.append({
            "case_id": case_id,
            "record_id": record_id,
            "audio_path": str(audio.resolve()),
            "audio_sha256": row.get("source_audio_sha256") or _sha256(audio),
            "start_s": start,
            "end_s": end,
            "reference": str(reference or ""),
            "hypothesis": str(hypothesis or ""),
            "provenance": provenance,
        })
    return cases


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def case_status(case_id: str, events: list[dict[str, Any]]) -> str:
    any_review = [event for event in events if event.get("case_id") == case_id and
                  event.get("event_type") in {"stage_a", "stage_a_correction", "stage_b", "stage_b_revision"}]
    reviews = [event for event in events if event.get("case_id") == case_id and
               event.get("event_type") in {"stage_b", "stage_b_revision"}]
    reviewer_ids = {event.get("reviewer_id") for event in reviews if event.get("reviewer_id")}
    if any(event.get("case_id") == case_id and event.get("event_type") == "adjudication" for event in events):
        return "adjudicated"
    if len(reviewer_ids) >= 2 and all(any(event.get("case_id") == case_id and
           event.get("reviewer_id") == reviewer and event.get("event_type") in {"stage_b", "stage_b_revision"}
           and event.get("independence_attested") for event in events) for reviewer in reviewer_ids):
        return "independently double-reviewed"
    if any_review:
        return "single-reviewer"
    return "unreviewed"


class ReviewApplication:
    def __init__(self, cases: list[dict[str, Any]], events_path: Path):
        self.cases = {case["case_id"]: case for case in cases}
        self.order = list(self.cases)
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def events(self) -> list[dict[str, Any]]:
        with self.lock:
            return read_events(self.events_path)

    def append(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self.lock, self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())

    def own_stage_a(self, case_id: str, reviewer_id: str) -> dict[str, Any] | None:
        history = self.stage_a_history(case_id, reviewer_id)
        return history[-1] if history else None

    def stage_a_history(self, case_id: str, reviewer_id: str) -> list[dict[str, Any]]:
        return [event for event in self.events() if event.get("case_id") == case_id and
                event.get("reviewer_id") == reviewer_id and event.get("event_type") in {"stage_a", "stage_a_correction"}]

    def neighboring_text_context(self, case_id: str) -> dict[str, Any]:
        """Return one adjacent source TextGrid interval on each side, never the reviewed interval."""
        from .data_inventory import _load_textgrid
        case = self.cases[case_id]
        source_value = case.get("provenance", {}).get("source_textgrid")
        if not source_value:
            raise FileNotFoundError("This case has no source TextGrid path for nearby text context")
        textgrid = Path(str(source_value))
        if not textgrid.is_absolute():
            textgrid = Path(case["audio_path"]).parent.parent / textgrid
        if not textgrid.is_file():
            raise FileNotFoundError("Source TextGrid is unavailable for nearby text context")
        rows = _load_textgrid(textgrid, {})
        previous = [(i, row) for i, row in enumerate(rows) if float(row["end_s"]) <= case["start_s"]]
        following = [(i, row) for i, row in enumerate(rows) if float(row["start_s"]) >= case["end_s"]]
        def item(pair: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
            index, row = pair
            return {"interval_id": "%s:interval:%06d" % (textgrid.stem, index),
                    "text": str(row.get("transcript", "")), "start_s": row.get("start_s"), "end_s": row.get("end_s")}
        return {"previous": item(previous[-1]) if previous else None,
                "next": item(following[0]) if following else None}


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Local speech review</title><style>
body{font:16px system-ui,sans-serif;max-width:850px;margin:2rem auto;padding:0 1rem;color:#182230}button,input,select,textarea{font:inherit;padding:.45rem;margin:.25rem}textarea{width:min(100%,42rem);min-height:5rem}fieldset{margin:1rem 0;border:1px solid #aab4c0}label{display:inline-block;margin:.35rem}small,.muted{color:#536273}.hidden{display:none}.case{font-size:1.4rem;font-weight:650}audio{width:min(100%,38rem)}
</style></head><body><h1>Medication speech review</h1><p class="muted">Local staged annotation. Hiding comparison text in Stage A reduces one source of priming but does not eliminate reviewer bias. These candidate sets were selected using source-reference medication strings, so selection itself is reference-informed; they are not random or exhaustive samples. No risk score or method ranking is shown.</p>
<label>Reviewer ID (non-identifying, e.g. reviewer-01) <input id="reviewer" pattern="reviewer-[A-Za-z0-9_-]{1,24}" required></label>
<label>Case <select id="case"></select></label><span id="caseid" class="case"></span>
<section id="stageA"><h2>Stage A — spoken content</h2><p>Listen first. Do not guess if the speech is unclear. Enter medication words as heard, one per line, or select a non-medication response.</p>
<audio id="audio" controls preload="none"></audio><div><button data-speed="0.75">0.75×</button><button data-speed="1">1×</button><button data-speed="1.25">1.25×</button><button data-speed="1.5">1.5×</button></div>
<label><input id="context" type="checkbox"> Include up to 3 seconds of nearby audio when replaying</label><br>
<label>What was spoken? <select id="stagea-kind"><option value="medication">One or more medication names</option><option value="no_medication">No medication</option><option value="unintelligible">Unintelligible</option><option value="uncertain">Uncertain / abstain</option></select></label><br>
<textarea id="stagea-text" placeholder="Medication name(s), as heard; one per line"></textarea><br>
<label>Uncertainty <select id="uncertainty"><option value="none_apparent">None apparent</option><option value="some">Some uncertainty</option><option value="high">High; abstain or flag</option></select></label>
<label>Notes <input id="stagea-note" size="55" placeholder="Optional; do not add patient-identifying details"></label><br>
<label><input id="independent" type="checkbox"> I completed Stage A without seeing another review of this case</label><br><button id="saveA">Save Stage A and reveal comparison</button><span id="asaved"></span></section>
<section id="stageB" class="hidden"><h2>Stage B — comparison</h2><p>Assess the reference and ASR separately. Ordinary review is not expert adjudication.</p><button id="showTextContext" disabled>Show adjacent source-text context (may include medication names)</button><p id="textContext" class="muted hidden"></p><h3>Source reference</h3><p id="reference"></p><h3>ASR hypothesis</h3><p id="hypothesis"></p>
<label>Reference fidelity <select id="fidelity"><option value="faithful">Faithful to audible speech</option><option value="questionable">Questionable / partially faithful</option><option value="reference_error">Contains a reference mistake</option><option value="cannot_assess">Cannot assess</option></select></label>
<fieldset><legend>ASR outcome (select all that apply)</legend><label><input type="checkbox" name="outcome" value="correct"> Correct</label><label><input type="checkbox" name="outcome" value="wrong_medication"> Wrong medication</label><label><input type="checkbox" name="outcome" value="non_medication_rendering"> Non-medication rendering</label><label><input type="checkbox" name="outcome" value="deletion"> Deletion</label><label><input type="checkbox" name="outcome" value="insertion"> Insertion</label><label><input type="checkbox" name="outcome" value="ambiguous"> Ambiguous</label></fieldset>
<label>Comparison notes <input id="stageb-note" size="65" placeholder="Reason for uncertainty, reference issue, multiple medicines, etc."></label><br><button id="saveB">Save comparison</button><span id="bsaved"></span>
<details><summary>Correct my original Stage A response</summary><p>Original saved response: <span id="originalStageA"></span></p><p>Correction is appended as a new event; the original response is retained.</p><label>Corrected spoken content <textarea id="corrected-text"></textarea></label><label>Reason for correction <input id="correction-reason" size="50"></label><button id="correctA">Append correction</button></details>
<section id="adjudication" class="hidden"><h3>Record a separate resolution (optional)</h3><p>This requires a third distinct reviewer ID after two independent completed reviews. It records a resolution, not expert credentials.</p><label>Adjudicator ID <input id="adjudicator" placeholder="reviewer-03"></label><label>Resolution <select id="resolution"><option value="correct">Correct</option><option value="wrong_medication">Wrong medication</option><option value="non_medication_rendering">Non-medication rendering</option><option value="deletion">Deletion</option><option value="insertion">Insertion</option><option value="ambiguous">Remain ambiguous</option><option value="reference_error">Reference error</option></select></label><label>Rationale <input id="resolution-note" size="50" required></label><button id="resolve">Append resolution</button></section></section>
<p id="message" role="status"></p><script>
const $=id=>document.getElementById(id), req=async(url,opt={})=>{const r=await fetch(url,opt),d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d};
let caseIds=[],contextActuallyPlayed=false,textContextActuallyShown=false,textContextIds=[]; async function init(){const d=await req('/api/cases');caseIds=d.cases;for(const id of caseIds){const o=document.createElement('option');o.value=id;o.textContent=id;$('case').append(o)}$('case').onchange=show;$('audio').addEventListener('play',()=>{if($('context').checked)contextActuallyPlayed=true});$('showTextContext').onclick=showTextContext;document.querySelectorAll('[data-speed]').forEach(b=>b.onclick=()=>{$('audio').playbackRate=Number(b.dataset.speed)});$('saveA').onclick=saveA;$('saveB').onclick=saveB;$('correctA').onclick=correctA;$('resolve').onclick=resolve;await show()}
function reviewer(){const id=$('reviewer').value.trim();if(!/^reviewer-[A-Za-z0-9_-]{1,24}$/.test(id))throw Error('Enter a non-identifying ID such as reviewer-01');return id}
async function show(){const id=$('case').value;if(!id)return;contextActuallyPlayed=false;textContextActuallyShown=false;textContextIds=[];$('textContext').textContent='';$('textContext').classList.add('hidden');$('showTextContext').disabled=true;$('caseid').textContent=id;$('stageB').classList.add('hidden');$('stageA').classList.remove('hidden');$('message').textContent='';$('audio').src='/api/audio/'+id+'?context='+($('context').checked?'1':'0');$('context').onchange=()=>{$('audio').src='/api/audio/'+id+'?context='+($('context').checked?'1':'0')};$('stagea-kind').value='medication';$('stagea-text').value='';$('uncertainty').value='some';$('stagea-note').value='';$('independent').checked=false;$('fidelity').value='faithful';$('stageb-note').value='';$('corrected-text').value='';$('correction-reason').value='';$('originalStageA').textContent='';document.querySelectorAll('input[name=outcome]').forEach(x=>x.checked=false);$('asaved').textContent='';$('bsaved').textContent='';const rid=$('reviewer').value;if(rid){try{const own=await req('/api/case/'+id+'?reviewer='+encodeURIComponent(reviewer()));if(own.stage_a){$('stagea-kind').value=own.stage_a.content_kind||'medication';$('stagea-text').value=own.stage_a.content_text||'';$('uncertainty').value=own.stage_a.uncertainty||'some';$('stagea-note').value=own.stage_a.notes||'';$('independent').checked=!!own.stage_a.independence_attested;$('corrected-text').value=own.stage_a.content_text||'';await reveal(id,rid,own.stage_a,own.stage_b,(own.stage_a_history||[])[0])} }catch(e){$('message').textContent=e.message}}}
async function showTextContext(){try{const d=await req('/api/context/'+$('case').value+'?reviewer='+encodeURIComponent(reviewer()));const bits=[];if(d.previous){bits.push('Before: '+d.previous.text);textContextIds.push(d.previous.interval_id)}if(d.next){bits.push('After: '+d.next.text);textContextIds.push(d.next.interval_id)}$('textContext').textContent=bits.join(' | ')||'No adjacent nonempty source intervals.';$('textContext').classList.remove('hidden');textContextActuallyShown=bits.length>0}catch(e){$('message').textContent=e.message}}
async function saveA(){try{const rid=reviewer(),id=$('case').value,kind=$('stagea-kind').value,text=$('stagea-text').value.trim();if(kind==='medication'&&!text)throw Error('Enter the medication name(s) heard or select another response');const payload={reviewer_id:rid,content_kind:kind,content_text:text,uncertainty:$('uncertainty').value,notes:$('stagea-note').value,context_used:contextActuallyPlayed||textContextActuallyShown,audio_context_used:contextActuallyPlayed,transcript_context_used:textContextActuallyShown,context_interval_ids:textContextIds,independence_attested:$('independent').checked};await req('/api/stage-a/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});$('asaved').textContent=' Saved.';await reveal(id,rid,payload,null)}catch(e){$('message').textContent=e.message}}
async function reveal(id,rid,stageA,stageB,original){const d=await req('/api/comparison/'+id+'?reviewer='+encodeURIComponent(rid));$('reference').textContent=d.reference||'(empty source reference)';$('hypothesis').textContent=d.hypothesis||'(empty ASR hypothesis)';const first=original||stageA;$('originalStageA').textContent=first.content_text||first.content_kind||'(empty)';if(stageB){$('fidelity').value=stageB.reference_fidelity||'faithful';$('stageb-note').value=stageB.notes||'';const saved=new Set(stageB.outcomes||[]);document.querySelectorAll('input[name=outcome]').forEach(x=>x.checked=saved.has(x.value));$('bsaved').textContent=' Previously saved comparison.'}$('showTextContext').disabled=false;$('stageB').classList.remove('hidden');$('stageA').classList.add('hidden');$('adjudication').classList.toggle('hidden',d.independent_reviewers<2)}
async function saveB(){try{const outcomes=[...document.querySelectorAll('input[name=outcome]:checked')].map(x=>x.value);if(!outcomes.length)throw Error('Select at least one ASR outcome');const d=await req('/api/stage-b/'+$('case').value,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer_id:reviewer(),outcomes,reference_fidelity:$('fidelity').value,notes:$('stageb-note').value,independence_attested:$('independent').checked,transcript_context_used:textContextActuallyShown,context_interval_ids:textContextIds})});$('bsaved').textContent=' Comparison saved.';$('adjudication').classList.toggle('hidden',d.review_status!=='independently double-reviewed')}catch(e){$('message').textContent=e.message}}
async function correctA(){try{const corrected=$('corrected-text').value.trim(),reason=$('correction-reason').value.trim();if(!reason)throw Error('Provide a correction reason');await req('/api/stage-a/'+$('case').value,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer_id:reviewer(),content_kind:$('stagea-kind').value,content_text:corrected,uncertainty:$('uncertainty').value,notes:$('stagea-note').value,context_used:contextActuallyPlayed||textContextActuallyShown,audio_context_used:contextActuallyPlayed,transcript_context_used:textContextActuallyShown,context_interval_ids:textContextIds,independence_attested:$('independent').checked,correction_reason:reason})});$('message').textContent='Stage A correction appended; original retained.'}catch(e){$('message').textContent=e.message}}
async function resolve(){try{const id=$('adjudicator').value.trim();if(!/^reviewer-[A-Za-z0-9_-]{1,24}$/.test(id))throw Error('Use a non-identifying reviewer ID');await req('/api/adjudicate/'+$('case').value,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer_id:id,resolution:$('resolution').value,rationale:$('resolution-note').value})});$('message').textContent='Resolution appended.'}catch(e){$('message').textContent=e.message}}
init().catch(e=>$('message').textContent=e.message);
</script></body></html>'''


def make_handler(app: ReviewApplication):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # Avoid writing source text or request query values into routine logs.
            print("review-http: " + (fmt % args).split("?")[0])

        def send_json(self, value: Mapping[str, Any], code: int = 200) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            host = self.headers.get("Host", "")
            if host and host != "localhost" and not host.startswith("127.0.0.1:"):
                self.send_json({"error": "loopback host only"}, 403)
                return
            parsed = urlparse(self.path)
            bits = parsed.path.strip("/").split("/")
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/cases":
                self.send_json({"cases": app.order})
                return
            if len(bits) == 3 and bits[:2] == ["api", "context"]:
                try:
                    self.case(bits[2])
                    reviewer = validate_reviewer(query.get("reviewer", [""])[0])
                    if not app.own_stage_a(bits[2], reviewer):
                        raise ValueError("Save Stage A before revealing adjacent source-text context")
                    self.send_json(app.neighboring_text_context(bits[2]))
                except (KeyError, ValueError, OSError, RuntimeError) as exc:
                    self.send_json({"error": str(exc)}, 400)
                return
            if len(bits) == 3 and bits[:2] == ["api", "audio"]:
                self.serve_audio(bits[2], query.get("context", ["0"])[0] == "1")
                return
            if len(bits) == 3 and bits[:2] == ["api", "case"]:
                case_id = bits[2]
                try:
                    self.case(case_id)
                    reviewer = validate_reviewer(query.get("reviewer", [""])[0])
                    own = app.own_stage_a(case_id, reviewer)
                    own_b = next((e for e in reversed(app.events()) if e.get("case_id") == case_id and
                                  e.get("reviewer_id") == reviewer and e.get("event_type") in {"stage_b", "stage_b_revision"}), None)
                    self.send_json({"stage_a": own, "stage_a_history": app.stage_a_history(case_id, reviewer), "stage_b": own_b})
                except (KeyError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, 400)
                return
            if len(bits) == 3 and bits[:2] == ["api", "comparison"]:
                case_id = bits[2]
                try:
                    reviewer = validate_reviewer(query.get("reviewer", [""])[0])
                    if not app.own_stage_a(case_id, reviewer):
                        raise ValueError("Save Stage A before revealing comparison")
                    case = self.case(case_id)
                    events = app.events()
                    independent = {e.get("reviewer_id") for e in events if e.get("case_id") == case_id and
                                   e.get("event_type") in {"stage_b", "stage_b_revision"} and e.get("independence_attested")}
                    self.send_json({"reference": case["reference"], "hypothesis": case["hypothesis"],
                                    "independent_reviewers": len(independent)})
                except (KeyError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, 400)
                return
            self.send_json({"error": "not found"}, 404)

        def case(self, case_id: str) -> dict[str, Any]:
            if case_id not in app.cases:
                raise KeyError("Unknown neutral case ID")
            return app.cases[case_id]

        def serve_audio(self, case_id: str, context: bool) -> None:
            try:
                case = self.case(case_id)
                payload = _audio_payload(case, context)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (KeyError, OSError, RuntimeError, ImportError) as exc:
                self.send_json({"error": str(exc)}, 404)

        def do_POST(self) -> None:  # noqa: N802
            host = self.headers.get("Host", "")
            if host and host != "localhost" and not host.startswith("127.0.0.1:"):
                self.send_json({"error": "loopback host only"}, 403)
                return
            parsed = urlparse(self.path)
            bits = parsed.path.strip("/").split("/")
            if len(bits) != 3 or bits[0] != "api":
                self.send_json({"error": "not found"}, 404)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                case = self.case(bits[2])
                reviewer = validate_reviewer(payload.get("reviewer_id", ""))
                prior_a = app.own_stage_a(case["case_id"], reviewer)
                audio_context_used = (bool(payload.get("audio_context_used")) if bits[1] == "stage-a"
                                      else bool(prior_a and prior_a.get("audio_context_used")))
                event: dict[str, Any] = {"case_id": case["case_id"], "reviewer_id": reviewer,
                                         "asset_binding": _event_asset_binding(case,
                                                                              audio_context_used=audio_context_used)}
                if bits[1] == "stage-a":
                    prior_a = app.own_stage_a(case["case_id"], reviewer)
                    if prior_a:
                        if not payload.get("correction_reason"):
                            raise ValueError("A Stage A correction requires a reason; the original response is retained")
                        event_type = "stage_a_correction"
                    else:
                        event_type = "stage_a"
                    kind = str(payload.get("content_kind", ""))
                    text = str(payload.get("content_text", ""))
                    if kind not in {"medication", "no_medication", "unintelligible", "uncertain"}:
                        raise ValueError("Invalid Stage A response")
                    if kind == "medication" and not text.strip():
                        raise ValueError("Enter medication words heard or select another Stage A response")
                    if payload.get("uncertainty") not in {"none_apparent", "some", "high"}:
                        raise ValueError("Record an uncertainty level")
                    event.update({"event_type": event_type, "content_kind": kind, "content_text": text,
                                  "uncertainty": str(payload.get("uncertainty", "")),
                                  "notes": str(payload.get("notes", "")),
                                  "context_used": bool(payload.get("context_used")),
                                  "audio_context_used": bool(payload.get("audio_context_used")),
                                  "transcript_context_used": bool(payload.get("transcript_context_used")),
                                  "context_interval_ids": list(payload.get("context_interval_ids", [])),
                                  "independence_attested": bool(payload.get("independence_attested")),
                                  "correction_reason": str(payload.get("correction_reason", ""))})
                elif bits[1] == "stage-b":
                    if not app.own_stage_a(case["case_id"], reviewer):
                        raise ValueError("Save Stage A before Stage B")
                    outcomes = payload.get("outcomes") or []
                    allowed = {"correct", "wrong_medication", "non_medication_rendering", "deletion", "insertion", "ambiguous"}
                    if not outcomes or set(outcomes) - allowed:
                        raise ValueError("Select one or more valid ASR outcomes")
                    if payload.get("reference_fidelity") not in {"faithful", "questionable", "reference_error", "cannot_assess"}:
                        raise ValueError("Select a reference-fidelity status")
                    event.update({"event_type": "stage_b" if not any(e.get("case_id") == case["case_id"] and
                                  e.get("reviewer_id") == reviewer and e.get("event_type") in {"stage_b", "stage_b_revision"}
                                  for e in app.events()) else "stage_b_revision",
                                  "outcomes": list(dict.fromkeys(outcomes)),
                                  "reference_fidelity": str(payload.get("reference_fidelity", "")),
                                  "notes": str(payload.get("notes", "")),
                                  "transcript_context_used": bool(payload.get("transcript_context_used")),
                                  "context_interval_ids": list(payload.get("context_interval_ids", [])),
                                  "independence_attested": bool(payload.get("independence_attested"))})
                elif bits[1] == "adjudicate":
                    events = app.events()
                    reviewers = {e.get("reviewer_id") for e in events if e.get("case_id") == case["case_id"] and
                                 e.get("event_type") in {"stage_b", "stage_b_revision"} and e.get("independence_attested")}
                    if len(reviewers) < 2 or reviewer in reviewers:
                        raise ValueError("A separate third reviewer is required after two independently attested reviews")
                    resolution = str(payload.get("resolution", ""))
                    if resolution not in {"correct", "wrong_medication", "non_medication_rendering", "deletion", "insertion", "ambiguous", "reference_error"}:
                        raise ValueError("Invalid resolution")
                    if not str(payload.get("rationale", "")).strip():
                        raise ValueError("A rationale is required")
                    event.update({"event_type": "adjudication", "resolution": resolution,
                                  "rationale": str(payload["rationale"])})
                else:
                    self.send_json({"error": "not found"}, 404)
                    return
                app.append(event)
                self.send_json({"saved": True, "review_status": case_status(case["case_id"], app.events())}, 201)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)


    return Handler


def validate_reviewer(value: Any) -> str:
    reviewer = str(value or "")
    if not re.fullmatch(r"reviewer-[A-Za-z0-9_-]{1,24}", reviewer):
        raise ValueError("Use a non-identifying reviewer ID such as reviewer-01")
    return reviewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Six-case JSON or development-prediction JSONL")
    parser.add_argument("--events", type=Path, required=True, help="Append-only local annotation JSONL")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    app = ReviewApplication(cases, args.events)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.daemon_threads = True
    url = "http://127.0.0.1:%d/" % args.port
    print(json.dumps({"status": "LOCAL_REVIEW_READY", "url": url, "cases": len(cases),
                      "events_path": str(args.events), "event_log_exists": args.events.exists(),
                      "audio_content_is_not_copied_to_a_bundle": True}, indent=2), flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
