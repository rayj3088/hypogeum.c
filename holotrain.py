#!/usr/bin/env python3
"""
holotrain.py - train HoloMoE to talk faster, for as long as you like.

  python3 holotrain.py                     run until you press Stop
  python3 holotrain.py 5m                  5 minutes   (also 90s, 30m, 2h, overnight; a bare number = minutes)
  python3 holotrain.py 2h --turbo          turbo = short fast rounds, precise = long careful rounds, auto = both
  python3 holotrain.py 30m --target 4      stop early once it talks at 4 tokens/s
  python3 holotrain.py --feed paper.pdf    pour any file (PDF, Word, Markdown, ePub, text...) into the guessing memory
  python3 holotrain.py 10m --model 2b      train another model (each model keeps its own learning)
  python3 holotrain.py 10m --no-browser    no window; progress lines in the terminal only
Everything can also be changed live in the window while it trains.

A window opens in your default browser with live stats, the ceilings for your Mac, and
Pause / Resume / Stop. What it learns is saved in the plate as it goes:
  plate/heat.bin    which experts to keep warm in RAM
  plate/draft.tok   guessing memory (word patterns it has seen and written)
  plate/tuned.cfg   the fastest settings it found (holomoe loads them at boot)
Stopping at any moment is safe: every file is written whole or not at all.
Standard library only.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "holomoe")
PLATE = os.path.expanduser(os.environ.get("HOLOMOE_PLATE", os.path.join(HERE, "plate")))
for _i, _a in enumerate(sys.argv):            # --model 30b / 2b: train that model's plate (35b is the default "plate")
    if _a == "--model" and _i + 1 < len(sys.argv):
        _m = sys.argv[_i + 1].lower()
        PLATE = os.path.join(os.path.dirname(PLATE), "plate" if _m == "35b" else "plate-" + _m)
        if not os.path.isdir(PLATE):
            sys.exit("no model %r yet - get it with: python3 shardpress.py all --model %s" % (_m, _m))
        del sys.argv[_i:_i + 2]
        break
TDIR = os.path.join(PLATE, "train")
STATUS = os.path.join(TDIR, "status.json")
CONTROL = os.path.join(TDIR, "control")

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>HoloMoE Training</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0b0f14;--card:#141b23;--ink:#e8eef5;--dim:#8a99a8;--acc:#5ad1ff;--good:#6ee7a0;--warn:#ffcf5a;--line:#243241}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:18px 16px 40px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--dim);margin-bottom:14px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.k{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.v{font-size:26px;font-weight:650;margin-top:2px}
.v small{font-size:13px;color:var(--dim);font-weight:400}.up{color:var(--good)}
.bar{height:12px;background:#0e141a;border-radius:6px;overflow:hidden;margin:6px 0 2px;border:1px solid var(--line)}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),var(--good))}
.cl{display:flex;justify-content:space-between;font-size:13px;color:var(--dim)}
button{background:#1d2833;color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:10px 18px;font-size:15px;margin-right:8px;cursor:pointer}
button:hover{border-color:var(--acc)}#stop{border-color:#7a3b3b}
.note{color:var(--warn);margin:8px 0}.txt{white-space:pre-wrap;color:#c9d6e2;font-size:13px;max-height:150px;overflow:auto}
svg{width:100%;height:190px;display:block}
.t,.m{margin-bottom:6px}.help{color:var(--dim);font-size:13px;margin-top:2px}
.drop{display:block;margin-top:8px;padding:22px;border:2px dashed var(--line);border-radius:12px;text-align:center;color:var(--dim);cursor:pointer}.drop.hot{border-color:var(--acc);color:var(--acc)}.on{border-color:var(--acc);background:#13324a}
.rec{border-color:#2b5b3f}
.phase{display:inline-block;padding:3px 10px;border-radius:20px;background:#13324a;color:var(--acc);font-size:13px}
</style></head><body><div class="wrap">
<h1>HoloMoE Training <span class="phase" id="phase">starting</span></h1>
<div class="sub">Qwen3.6-35B-A3B on your Mac &mdash; learning which experts to keep warm, what words come next, and the fastest settings.</div>
<div style="margin:6px 0 10px"><button id="pause">Pause</button><button id="resume">Resume</button><button id="stop">Stop &amp; save</button></div>
<div class="card rec" style="margin-bottom:10px">
 <div class="k">Record to beat &mdash; ByteShape, Qwen3-30B on Raspberry Pi 5 (16 GB RAM, model fits in RAM)</div>
 <div class="row" style="margin:8px 0 0">
  <div><div class="k">Speed</div><div class="v"><span id="r_you">&ndash;</span> <small>/ 8.03 tok/s</small></div><div class="bar"><i id="r_bar" style="width:0%"></i></div></div>
  <div><div class="k">Quality</div><div class="v">not measured <small>/ 94.18%</small></div><div class="help">needs the quality test (coming)</div></div>
  <div><div class="k">Hardware</div><div class="v up">&#10003; <small>8 GB, 35B, 2015 dual-core vs 16 GB, 30B</small></div></div>
 </div>
</div>
<div class="card" style="margin-bottom:10px">
 <div class="k">Live &mdash; this practice round</div>
 <div class="row" style="margin:8px 0 4px">
  <div><div class="k">Speed this round</div><div class="v" id="l_tps">&ndash;</div></div>
  <div><div class="k">Tokens written</div><div class="v" id="l_tok">&ndash;</div></div>
  <div><div class="k">Guesses landing</div><div class="v" id="l_acc">&ndash;</div></div>
 </div>
 <div class="txt" id="l_text" style="max-height:90px"></div>
</div>
<div class="card" style="margin-bottom:10px">
 <div class="k">How long to train</div>
 <div class="help">Counts from the moment you click. You can change it any time.</div>
 <div style="margin:6px 0 12px"><button class="t" data-s="300">5 min</button><button class="t" data-s="1800">30 min</button><button class="t" data-s="7200">2 hours</button><button class="t" data-s="28800">Overnight</button><button class="t" data-s="0">Until I stop it</button></div>
 <div class="k">Mode</div>
 <div style="margin:6px 0 4px"><button class="m" data-m="auto">Auto &#9733; best</button><button class="m" data-m="turbo">Turbo</button><button class="m" data-m="precise">Precise</button></div>
 <div class="help" id="modeinfo">Auto starts in turbo, then switches to careful measuring so the upgrades it keeps are real.</div>
 <div class="k">Goal speed (optional)</div>
 <div class="help">Training stops by itself the moment it talks this fast. Leave blank to keep going until the time runs out.</div>
 <div style="margin:6px 0 12px"><input id="target" type="number" step="0.5" min="0" placeholder="e.g. 3" style="background:#0e141a;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px;width:110px"> <span class="help" style="display:inline">tokens / second</span> <button id="settarget">Set</button></div>
 <div class="k">Feed it anything</div>
 <div class="help">PDF, Word, Markdown, text, web pages, ePub books, PowerPoint, Excel, code, and images (JPG/PNG/HEIC - it reads the labels and text in them). Each file goes into its guessing memory in seconds and becomes practice questions, so training focuses on your topics.</div>
 <label id="drop" class="drop"><input type="file" id="feed" multiple style="display:none">Drop files here or click to choose</label>
 <div id="fedlist" class="help"></div>
</div>
<div class="row">
 <div class="card"><div class="k">Talking speed (best settings, this answer length)</div><div class="v" id="best">&ndash;</div></div>
 <div class="card"><div class="k">Started at (same answer length)</div><div class="v" id="first">&ndash;</div></div>
 <div class="card"><div class="k">Gain so far</div><div class="v up" id="gain">&ndash;</div></div>
 <div class="card"><div class="k">Round</div><div class="v" id="round">&ndash;</div></div>
</div>
<div class="row">
 <div class="card"><div class="k">Guesses landing</div><div class="v" id="acc">&ndash;</div></div>
 <div class="card"><div class="k">Tokens per pass</div><div class="v" id="wpp">&ndash;</div></div>
 <div class="card"><div class="k">Experts already in RAM</div><div class="v" id="hits">&ndash;</div></div>
 <div class="card"><div class="k">Quality (perplexity)</div><div class="v" id="ppl">&ndash;</div></div>
</div>
<div class="card" style="margin-bottom:10px"><div class="k">Tokens per second, round by round</div><svg id="chart" viewBox="0 0 600 190" preserveAspectRatio="none"></svg>
<div class="cl"><span>&#9679; <span style="color:var(--acc)">best settings</span> &nbsp; &#9679; <span style="color:var(--warn)">experiments</span></span><span id="elapsed"></span></div></div>
<div class="card" style="margin-bottom:10px"><div class="k">Theoretical maximums on this Mac</div>
 <div id="ceil"></div>
 <div class="cl" style="margin-top:6px"><span id="hw"></span></div></div>
<div class="row">
 <div class="card"><div class="k">Best settings</div><div id="set" style="margin-top:6px"></div></div>
 <div class="card"><div class="k">What it has learned</div><div id="learn" style="margin-top:6px"></div></div>
</div>
<div class="note" id="note"></div>
<div class="card"><div class="k">Latest practice question and answer</div><div class="txt" id="text"></div></div>
</div><script>
const $=id=>document.getElementById(id), f=(x,d=2)=>x==null?'–':(+x).toFixed(d);
function post(q){return fetch('/control?'+q,{method:'POST'})}
$('pause').onclick=()=>post('cmd=pause');$('resume').onclick=()=>post('cmd=run');$('stop').onclick=()=>post('cmd=stop');
document.querySelectorAll('.t').forEach(b=>b.onclick=()=>post('more='+b.dataset.s));
document.querySelectorAll('.m').forEach(b=>b.onclick=()=>post('mode='+b.dataset.m));
$('settarget').onclick=()=>post('target='+(+$('target').value||0));
async function sendFiles(files){for(const f of files){const li=document.createElement('div');li.textContent=f.name+' - reading...';$('fedlist').prepend(li);
 try{const r=await (await fetch('/feed?name='+encodeURIComponent(f.name),{method:'POST',body:await f.arrayBuffer()})).json();
  li.innerHTML=r.ok?`&#10003; <b>${r.name}</b> - ${r.words.toLocaleString()} words (${r.how}) fed, ${r.questions} practice questions written`:`&#10007; <b>${r.name}</b> - ${r.error}`}catch(e){li.textContent=f.name+' - failed'}}}
$('feed').onchange=()=>{sendFiles([...$('feed').files]);$('feed').value=''};
['dragenter','dragover'].forEach(e=>$('drop').addEventListener(e,ev=>{ev.preventDefault();$('drop').classList.add('hot')}));
['dragleave','drop'].forEach(e=>$('drop').addEventListener(e,ev=>{ev.preventDefault();$('drop').classList.remove('hot')}));
$('drop').addEventListener('drop',ev=>sendFiles([...ev.dataTransfer.files]));
const MI={auto:'Auto: starts in turbo, then switches to careful measuring so the upgrades it keeps are real.',turbo:'Turbo: short rounds, the most ideas tested per minute. Timing is rougher, so it only keeps clear wins.',precise:'Precise: long rounds, exact timing. Best for locking in settings at the end.'};
function bar(label,val,max,top){const w=Math.max(1,Math.min(100,100*val/top));
 return `<div class="cl"><span>${label}</span><span>${f(val)} tok/s</span></div><div class="bar"><i style="width:${w}%"></i></div>`}
function chart(hb,hc,top){let s='',n=hb.length;if(!n){$('chart').innerHTML='';return}
 const mx=Math.max(top||0,...hb,...hc)*1.1||1,X=i=>n<2?300:i*600/(n-1),Y=v=>186-v/mx*180;
 let p='';hb.forEach((v,i)=>{if(v>=0)p+=(p?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)});
 s+=`<path d="${p}" fill="none" stroke="#5ad1ff" stroke-width="2.5"/>`;
 hc.forEach((v,i)=>{if(v>=0)s+=`<circle cx="${X(i)}" cy="${Y(v)}" r="3.5" fill="#ffcf5a"/>`});
 if(top)s+=`<line x1="0" x2="600" y1="${Y(top)}" y2="${Y(top)}" stroke="#6ee7a0" stroke-dasharray="5 5"/><text x="4" y="${Y(top)-4}" fill="#6ee7a0" font-size="11">one-word ceiling ${f(top)}</text>`;
 $('chart').innerHTML=s}
async function tick(){try{const st=await (await fetch('/state')).json();
 document.querySelectorAll('.m').forEach(b=>b.classList.toggle('on',b.dataset.m==st.mode));
 const d=await (await fetch('/status')).json();
 $('phase').textContent=d.phase||'';
 if(d.wps_best!=null){$('best').innerHTML=f(d.wps_best)+' <small>tok/s</small>';$('r_you').textContent=f(d.wps_top||d.wps_best);$('r_bar').style.width=Math.min(100,100*(d.wps_top||d.wps_best)/8.03)+'%';}
 if(d.wps_first){$('first').innerHTML=f(d.wps_first)+' <small>tok/s</small>';const g=100*(d.wps_best/d.wps_first-1);$('gain').textContent=(g>=0?'+':'')+f(g,0)+'%'}
 if(d.round!=null)$('round').textContent=d.round;
 if(d.accept!=null){$('acc').textContent=f(100*d.accept,0)+'%';$('wpp').textContent=f(d.wpp);$('hits').textContent=f(100*d.hits,0)+'%'}
 if(d.ppl){$('ppl').innerHTML=f(d.ppl,2)+' <small>start '+f(d.ppl0,2)+'</small>'}
 if(d.ceil_mem){const top=Math.max(d.ceil_guess,d.ceil_mem,d.ceil_cpu);
  $('ceil').innerHTML=bar('Your speed now',d.wps_best||0,0,top)+bar('Memory ceiling, one token per pass',d.ceil_mem,0,top)+
   bar('Compute ceiling, one token per pass',d.ceil_cpu,0,top)+bar('Record to beat (ByteShape Pi 5)',8.03,0,top)+bar('Perfect-guessing ceiling (9 tokens per pass)',d.ceil_guess,0,top);
  if(d.bw)$('hw').textContent=`measured: memory ${f(d.bw,1)} GB/s, int8 math ${f(d.ops,1)} billion multiply-adds/s`;}
 if(d.hb)chart(d.hb,d.hc,d.ceil_mem&&d.ceil_mem<4*(d.wps_top||1)?Math.min(d.ceil_mem,d.ceil_cpu):0);
 if(d.draft_k!=null)$('set').innerHTML=`guess <b>${d.draft_k}</b> words ahead<br>skip experts under <b>${f(100*d.min_share,0)}%</b> share<br><span style="color:var(--dim)">${d.kept||0} upgrades kept of ${d.tried||0} tried</span>`;
 if(d.corpus!=null)$('learn').innerHTML=`guessing memory: <b>${d.corpus.toLocaleString()}</b> words<br>experts mapped: <b>${d.woken}</b> of ${d.experts}`;
 if(d.elapsed!=null)$('elapsed').textContent=Math.floor(d.elapsed/60)+' min'+(d.limit>0?' of '+Math.round(d.limit/60):' (no limit)');
 $('modeinfo').textContent=MI[st.mode]+(d.round_len?' It picks the answer length itself: now '+d.round_len+' tokens per round.':'');
 if(d.fed)$('learn').innerHTML+=`<br>fed from your files: <b>${d.fed.toLocaleString()}</b> words`;
 if(d.note)$('note').textContent=d.note; if(d.text)$('text').textContent=d.text;
 if(d.phase=='done')$('note').textContent='Training finished and saved. Your next ./holomoe boots with these settings.';
}catch(e){} setTimeout(tick,3000)} tick();
async function live(){try{const d=await (await fetch('/live')).json();
 if(d.tps!=null){$('l_tps').innerHTML=f(d.tps)+' <small>tok/s</small>';$('l_tok').innerHTML=d.tokens+' <small>of '+d.limit+'</small>';$('l_acc').textContent=f(100*d.accept,0)+'%';$('l_text').textContent=d.text||''}
}catch(e){} setTimeout(live,3000)} live();
</script></body></html>"""


STATE = {"state": "run", "limit": 0.0, "target": 0.0, "round": 0, "mode": "auto", "feeds": []}
LOCK = threading.Lock()


def write_control():
    with LOCK:
        lines = ["state=" + STATE["state"], "limit=%d" % STATE["limit"], "target=%g" % STATE["target"],
                 "round=%d" % STATE["round"], "mode=" + STATE["mode"]] + ["feed=" + p for p in STATE["feeds"]]
        tmp = CONTROL + ".part"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, CONTROL)


def elapsed():
    try:
        with open(STATUS) as f:
            return float(json.load(f).get("elapsed", 0))
    except (OSError, ValueError):
        return 0.0


TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".tex", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".log", ".ini",
            ".cfg", ".toml", ".py", ".c", ".h", ".cc", ".cpp", ".hpp", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
            ".swift", ".m", ".sh", ".zsh", ".rb", ".php", ".css", ".sql", ".r", ".jl", ".lua", ".pl", ".kt", ".srt", ".vtt"}


def strip_tags(t):
    import html, re
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return html.unescape(t)


def zip_text(path, pick):
    import zipfile, re
    out = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist(), key=lambda x: [int(d) if d.isdigit() else d for d in re.split(r"(\d+)", x)]):
            if pick(n):
                out.append(strip_tags(z.read(n).decode("utf-8", "replace")))
    return "\n".join(out)


def pdf_text(path):
    """macOS PDFKit through JavaScript for Automation (built into every Mac), else a plain-Python fallback"""
    if sys.platform == "darwin":
        js = ('ObjC.import("Quartz");function run(a){var d=$.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(a[0]));'
              'if(d.isNil())return "";var s=d.string;return s.isNil()?"":s.js}')
        try:
            r = subprocess.run(["osascript", "-l", "JavaScript", "-e", js, path], capture_output=True, timeout=300)
            t = r.stdout.decode("utf-8", "replace")
            if t.strip():
                return t, "PDF via macOS"
        except (OSError, subprocess.TimeoutExpired):
            pass
    import re, zlib
    raw = open(path, "rb").read(); parts = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        data = m.group(1)
        try:
            data = zlib.decompress(data)
        except zlib.error:
            pass
        for tj in re.finditer(rb"\[(.*?)\]\s*TJ|\((.*?)(?<!\\)\)\s*Tj", data, re.S):
            seg = tj.group(1) if tj.group(1) is not None else b"(" + tj.group(2) + b")"
            parts.append(b"".join(re.findall(rb"\((.*?)(?<!\\)\)", seg, re.S)).replace(b"\\(", b"(").replace(b"\\)", b")"))
        if re.search(rb"T\*|Td|TD|ET", data):
            parts.append(b"\n")
    return b" ".join(parts).decode("latin-1"), "PDF"


IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".gif", ".bmp", ".webp"}


def image_text(path):
    """macOS Vision text recognition (built into every Mac since 10.15): labels, part numbers, notes, captions"""
    if sys.platform != "darwin":
        raise ValueError("reading text from images needs macOS")
    js = r"""ObjC.import("Vision");
function run(a){var url=$.NSURL.fileURLWithPath(a[0]);var req=$.VNRecognizeTextRequest.alloc.init;
req.setRecognitionLevel(0);req.setUsesLanguageCorrection(true);
var h=$.VNImageRequestHandler.alloc.initWithURLOptions(url,$({}));
if(!h.performRequestsError($([req]),null))return "";var r=req.results,out=[];
for(var i=0;i<r.count;i++){var c=r.objectAtIndex(i).topCandidates(1);if(c.count>0)out.push(c.objectAtIndex(0).string.js)}
return out.join("\n")}"""
    try:
        r = subprocess.run(["osascript", "-l", "JavaScript", "-e", js, path], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("image reader did not respond")
    t = r.stdout.decode("utf-8", "replace")
    if not t.strip():
        raise ValueError("no readable text in this image")
    return t


def extract(path):
    """any file -> plain text. Returns (text, how) or raises ValueError."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXT:
        t, how = image_text(path), "text read from image"
        if len(t.split()) < 1:
            raise ValueError("no readable text in this image")
        return t.strip(), how
    if ext == ".pdf":
        t, how = pdf_text(path)
    elif ext in (".docx", ".docm", ".dotx"):
        t, how = zip_text(path, lambda n: n.startswith("word/") and n.endswith(".xml") and ("document" in n or "footnotes" in n)), "Word"
    elif ext in (".pptx", ".pptm"):
        t, how = zip_text(path, lambda n: n.startswith("ppt/slides/slide") and n.endswith(".xml")), "PowerPoint"
    elif ext in (".xlsx", ".xlsm"):
        t, how = zip_text(path, lambda n: n == "xl/sharedStrings.xml"), "Excel"
    elif ext in (".odt", ".odp", ".ods"):
        t, how = zip_text(path, lambda n: n == "content.xml"), "OpenDocument"
    elif ext == ".epub":
        t, how = zip_text(path, lambda n: n.lower().endswith((".xhtml", ".html", ".htm"))), "ePub"
    elif ext in (".doc", ".rtf", ".rtfd", ".wordml", ".webarchive") and sys.platform == "darwin":
        r = subprocess.run(["textutil", "-convert", "txt", "-stdout", path], capture_output=True)
        t, how = r.stdout.decode("utf-8", "replace"), ext[1:].upper()
    elif ext in (".html", ".htm", ".xhtml"):
        t, how = strip_tags(open(path, "rb").read().decode("utf-8", "replace")), "web page"
    else:
        raw = open(path, "rb").read()
        if ext not in TEXT_EXT:
            sample = raw[:65536]
            if b"\0" in sample or sum(32 <= b < 127 or b in (9, 10, 13) or b >= 128 for b in sample) < 0.9 * max(1, len(sample)):
                raise ValueError("not a text-like file (try saving it as PDF or text)")
        t, how = raw.decode("utf-8", "replace"), "text"
    import re
    t = re.sub(r"[ \t\f\v]+", " ", t); t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t).strip()
    if len(t.split()) < 5:
        raise ValueError("no readable text found (a scanned image?)")
    return t, how


def write_prompts(text, name):
    """turn a fed file into practice questions about its own topics (train/prompts.txt, newest kept)"""
    import random, re
    out = []
    heads = [h.strip("#*= ").strip() for h in re.findall(r"(?m)^\s*#{1,4}\s+(.{4,80})$", text)]
    heads += [l.strip() for l in text.splitlines() if 3 <= len(l.split()) <= 9 and l.strip()[:1].isupper()
              and not l.strip().endswith((".", ",", ";", ":")) and sum(c.isalpha() for c in l) > 0.7 * len(l.strip())]
    for h in list(dict.fromkeys(heads))[:40]:
        out.append(random.choice(["Explain %s in plain language.", "What are the key ideas behind %s?",
                                  "Give a short overview of %s.", "How would you teach %s to a beginner?"]) % h)
    plain = re.sub(r"[#*_`>|]+", " ", text)
    sents = list(dict.fromkeys(x.strip() for x in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", plain)) if 12 <= len(x.split()) <= 40))
    random.shuffle(sents)
    for x in sents[:60]:
        out.append(random.choice(["Explain this in simple terms: %s", "Expand on this idea: %s",
                                  "Continue this explanation: %s", "Is this accurate? Explain: %s"]) % x)
    code = re.findall(r"(?m)^\s*(?:def|class|static|int|void|float|function|fn|func)\s+(\w+)", text)
    for c in list(dict.fromkeys(code))[:20]:
        out.append("Explain what a function called %s probably does, with an example." % c)
    if not out:
        return 0
    path = os.path.join(TDIR, "prompts.txt")
    old = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            old = [l.rstrip("\n") for l in f if l.strip()]
    lines = (old + [o.replace("\n", " ")[:600] for o in out])[-3000:]
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return len(out)


def feed_any(src, name=None):
    """extract text from any file, save it next to the training data, queue it for the engine"""
    name = os.path.basename(name or src)
    t, how = extract(src)
    d = os.path.join(TDIR, "feeds"); os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.splitext(name)[0] + ".txt")
    k = 2
    while os.path.exists(dst):
        dst = os.path.join(d, "%s-%d.txt" % (os.path.splitext(name)[0], k)); k += 1
    with open(dst, "w", encoding="utf-8") as f:
        f.write(t)
    add_feed(dst)
    q = write_prompts(t, name)
    return {"ok": True, "name": name, "words": len(t.split()), "how": how, "questions": q}


def add_feed(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path) and path not in STATE["feeds"]:
        STATE["feeds"].append(path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            body = json.dumps(STATE).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/status") or self.path.startswith("/live"):
            try:
                with open(STATUS if self.path.startswith("/status") else os.path.join(TDIR, "live.json"), "rb") as f:
                    body = f.read()
            except OSError:
                body = b'{"phase":"starting"}'
            self.send_response(200); self.send_header("Content-Type", "application/json")
        else:
            body = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        from urllib.parse import urlparse, parse_qs, unquote
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/feed":                       # a file dropped in the window: pull out its text, then feed it
            import tempfile
            n = int(self.headers.get("Content-Length", 0)); data = self.rfile.read(n)
            name = os.path.basename(unquote(q.get("name", "file.txt"))) or "file.txt"
            fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(name)[1])
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            try:
                res = feed_any(tmp, name)
            except Exception as e:                  # report, never crash the window
                res = {"ok": False, "name": name, "error": str(e)}
            finally:
                os.unlink(tmp)
            write_control()
            body = json.dumps(res).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)
            return
        else:
            if q.get("cmd") in ("run", "pause", "stop"): STATE["state"] = q["cmd"]
            if "more" in q:
                m = float(q["more"]); STATE["limit"] = elapsed() + m if m > 0 else 0
            if q.get("mode") in ("auto", "turbo", "precise"): STATE["mode"] = q["mode"]
            if "round" in q: STATE["round"] = max(0, min(256, int(float(q["round"]))))
            if "target" in q: STATE["target"] = max(0.0, float(q["target"]))
        write_control()
        self.send_response(204); self.end_headers()


def duration(a):
    a = a.strip().lower()
    if a in ("overnight", "night"): return 8 * 3600
    for suf, mul in (("s", 1), ("sec", 1), ("m", 60), ("min", 60), ("h", 3600), ("hr", 3600), ("hours", 3600)):
        if a.endswith(suf) and a[: -len(suf)].replace(".", "", 1).isdigit():
            return float(a[: -len(suf)]) * mul
    return float(a) * 60                            # bare number = minutes


HELP = """holotrain.py - train HoloMoE to talk faster. A window opens in your default browser with live stats.

  python3 holotrain.py [TIME] [MODE] [options]

TIME     90s  5m  10m  30m  2h  overnight  (a bare number = minutes; none = until you press Stop)
MODE     --auto     default: quick rounds first, then careful ones to lock in real wins
         --turbo    short rounds, most settings tried per minute (best for 5-10 minute runs)
         --precise  long rounds, most accurate timing
OPTIONS  --target 3          stop as soon as it talks at 3 tok/s
         --feed FILE         feed a file first (repeatable): PDF, Word, Markdown, text, web pages,
                             ePub, PowerPoint, Excel, code, images (reads their text)
         --round N           force practice answer length in tokens (normally automatic)
         --model NAME        train another model's folder (35b default, 2b, 30b)
         --no-browser        don't open the window; progress is printed in the terminal either way

WHAT IT DOES   each round answers a practice question (from your fed files + built-in topics),
               times it, and tries a new setting every other round: how many words to guess
               ahead, and skipping faint experts - kept only if faster AND quality stays
               within 2%. Learning saves into the plate as it goes; Stop is always safe.
THE WINDOW     record scoreboard (vs 8.03 tok/s Pi 5 record), live round, speed chart, your
               Mac's measured ceilings, time buttons, mode, goal speed, drag-and-drop feeding.
"""


def main():
    args = sys.argv[1:]; i = 0
    if any(a in ("-h", "--help") for a in args):
        print(HELP); return
    while i < len(args):
        a = args[i]
        if a in ("--turbo", "--precise", "--auto"): STATE["mode"] = a[2:]
        elif a == "--round" and i + 1 < len(args): STATE["round"] = int(args[i + 1]); i += 1
        elif a == "--no-browser": pass                  # print progress here only; don't open a window
        elif a == "--target" and i + 1 < len(args): STATE["target"] = float(args[i + 1]); i += 1
        elif a == "--feed" and i + 1 < len(args):
            try:
                os.makedirs(TDIR, exist_ok=True); r = feed_any(os.path.expanduser(args[i + 1]))
                print("feeding %s: %d words (%s), %d practice questions" % (r["name"], r["words"], r["how"], r["questions"]))
            except Exception as e:
                print("could not feed %s: %s" % (args[i + 1], e))
            i += 1
        elif not a.startswith("-"): STATE["limit"] = duration(a)
        else: sys.exit("unknown option " + a)
        i += 1
    if not os.path.exists(ENGINE):
        sys.exit("holomoe is not built yet - run: make -f holomoe.mk holomoe")
    os.makedirs(TDIR, exist_ok=True)
    for p in (STATUS,):
        if os.path.exists(p):
            os.remove(p)
    write_control()
    srv = None
    for port in range(8766, 8790):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler); break
        except OSError:
            continue
    if not srv:
        sys.exit("no free port for the status window")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/" % port
    print("training window: %s  (any browser; progress is also printed here)" % url)
    if "--no-browser" not in sys.argv:
        try:
            import webbrowser
            webbrowser.open(url)                      # the system's default browser, whatever it is
        except Exception:
            pass
    log = open(os.path.join(TDIR, "engine.log"), "w")
    proc = subprocess.Popen([ENGINE, PLATE, "--train", "0"], stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)      # Ctrl-C here stops cleanly after the round instead of killing mid-save
    print("limit: %s, mode: %s" % (("%.0f min" % (STATE["limit"] / 60) if STATE["limit"] >= 60 else "%.0f s" % STATE["limit"]) if STATE["limit"] else "until you press Stop", STATE["mode"]))

    def stop(*_):
        STATE["state"] = "stop"; write_control()
        print("\nstopping after this round (saving)...")
    signal.signal(signal.SIGINT, stop)
    last_line, last_t = "", 0.0
    while proc.poll() is None:
        time.sleep(1)
        if time.time() - last_t < 10:
            continue
        last_t = time.time()
        try:
            with open(STATUS) as f:
                st = json.load(f)
        except (OSError, ValueError):
            continue
        bits = [str(st.get("phase", ""))]
        if st.get("trial") is not None:
            bits.append("round %s" % st["trial"])
        if st.get("wps_now"):
            bits.append("now %.2f words/s" % float(st["wps_now"]))
        if st.get("wps_best"):
            bits.append("best %.2f words/s" % float(st["wps_best"]))
        line = " | ".join(b for b in bits if b)
        if line and line != last_line:                 # plain progress line for the terminal
            last_line = line
            print("  [%s] %s" % (time.strftime("%H:%M"), line), flush=True)
    print("training finished - settings and learning saved in", PLATE)
    time.sleep(3)   # let the window show the final numbers
    srv.shutdown()


if __name__ == "__main__":
    main()
