#!/usr/bin/env python3
"""
shardpress - download a big model one shard at a time, convert each shard into
a HoloMoE plate, delete the raw shard, repeat. Pause/resume any time.

  python3 shardpress.py models              # the menu: which models, how big
  python3 shardpress.py all --model 2b      # everything that's left, for one model
  python3 shardpress.py test                # quick real test on one shard
  python3 shardpress.py 1-7                 # a session: shards 1..7
  python3 shardpress.py status              # print progress and exit

--model picks the model (default 35b). Each model gets its own plate folder,
so they never mix: 35b -> ~/holomoe/plate, 30b -> ~/holomoe/plate-30b, 2b -> ~/holomoe/plate-2b.
When the last shard is done, the engine's model.cfg is written automatically.

A status page opens in your default browser with Pause / Resume / Stop, and the same
progress is printed in the terminal (--no-browser skips the page).
Safe to quit at any moment (Ctrl-C, closing Terminal, power loss): downloads
resume mid-file, converted pieces are atomic and checksummed, and a raw shard
is deleted only after every piece made from it is verified.
Standard library only. Everything lives in ~/holomoe/.
"""
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = {   # name: (repo, plate folder, one line for the menu)
    "35b": ("Qwen/Qwen3.6-35B-A3B", "plate", "Qwen3.6-35B-A3B   35B MoE, 3B per word   ~67 GB download, ~14 GB plate"),
    "30b": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "plate-30b", "Qwen3-30B-A3B-2507 30B MoE, 3B per word   ~61 GB download, ~11 GB plate  (untested on the reference build)"),
    "2b": ("Qwen/Qwen3.5-2B", "plate-2b", "Qwen3.5-2B         2B dense, all in RAM    ~4.6 GB download, ~4.5 GB plate"),
}
MODEL = "35b"
for _i, _a in enumerate(sys.argv):
    if _a == "--model" and _i + 1 < len(sys.argv):
        MODEL = sys.argv[_i + 1].lower()
    elif _a.startswith("--model="):
        MODEL = _a.split("=", 1)[1].lower()
if MODEL not in MODELS:
    sys.exit("unknown model %r - choose one of: %s" % (MODEL, ", ".join(MODELS)))
REPO, PLATE_DIR, LABEL = MODELS[MODEL]
SHARDS = []                         # filled from the model's index file
TEST_SHARD = 7 if MODEL == "35b" else 0   # 35b: ~1.1 GB shard with experts; others: the last (smallest) shard
META_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "model.safetensors.index.json", "vocab.json", "merges.txt"]
MARGIN = int(1.5 * 1024 ** 3)       # always leave this much disk free

ROOT = os.path.expanduser(os.environ.get("HOLOMOE_HOME", "~/holomoe"))
RAW, PLATE = os.path.join(ROOT, "raw"), os.path.join(ROOT, PLATE_DIR)
META = os.path.join(PLATE, "meta")
STATE = os.path.join(ROOT, "state.json" if MODEL == "35b" else "state-%s.json" % MODEL)
HERE = os.path.dirname(os.path.abspath(__file__))
HOLOPRESS = os.path.join(HERE, "holopress")


def shard_name(i):
    return SHARDS[i - 1]


def nshards():
    return len(SHARDS)


# ------------------------------------------------------------------ shared status

class Status:
    def __init__(self):
        self.lock = threading.Lock()
        self.d = {"phase": "starting", "shard": 0, "range": [], "done": [], "bytes_now": 0, "bytes_total": 0,
                  "speed": 0.0, "pieces": "", "free_gb": 0.0, "paused": False, "log": [], "finished": False,
                  "overall_done": 0, "overall_total": 0, "nshards": 0}
        self.pause = threading.Event()
        self.stop = threading.Event()
        self.proc = None

    def set(self, **kw):
        with self.lock:
            self.d.update(kw)

    def log(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with self.lock:
            self.d["log"] = (self.d["log"] + [line])[-12:]

    def snapshot(self):
        with self.lock:
            s = dict(self.d)
        s["paused"] = self.pause.is_set()
        s["free_gb"] = round(shutil.disk_usage(ROOT).free / 1024 ** 3, 1)
        return s

    def interrupt(self):
        p = self.proc
        if p and p.poll() is None:
            p.terminate()


ST = Status()
TEST_CONV_PAUSE = threading.Event()     # self-test: pause right after the first converted piece

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>HoloMoE shardpress</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#171a21;--fg:#e8eaf0;--mut:#8b93a7;--acc:#7cc4ff;--ok:#5bd69b;--warn:#f2c14e}
body{margin:0;background:var(--bg);color:var(--fg);font:15px -apple-system,system-ui,sans-serif;padding:16px}
.c{max-width:640px;margin:auto}.card{background:var(--card);border-radius:14px;padding:18px;margin-bottom:14px}
h1{font-size:19px;margin:0 0 4px}.m{color:var(--mut);font-size:13px}
.bar{height:14px;background:#262b36;border-radius:7px;overflow:hidden;margin:8px 0}
.fill{height:100%;background:var(--acc);transition:width .4s}.fill.o{background:var(--ok)}
.row{display:flex;justify-content:space-between;font-size:13px;color:var(--mut)}
button{font:600 15px system-ui;border:0;border-radius:10px;padding:10px 18px;margin-right:8px;cursor:pointer}
#pz{background:var(--warn);color:#111}#rs{background:var(--ok);color:#111}#sp{background:#3a404d;color:var(--fg)}
pre{white-space:pre-wrap;font-size:12px;color:var(--mut);margin:0}.big{font-size:22px;font-weight:700}
</style></head><body><div class="c">
<div class="card"><h1>HoloMoE &middot; __REPO__</h1><div class="m" id="ph">connecting...</div></div>
<div class="card"><div class="row"><span>Whole job</span><span id="ot"></span></div>
<div class="bar"><div class="fill o" id="ob" style="width:0"></div></div>
<div class="row"><span>Shards done: <b id="sd"></b></span><span>Free disk: <b id="fd"></b> GB</span></div></div>
<div class="card"><div class="row"><span id="cs">Current shard</span><span id="sp2"></span></div>
<div class="bar"><div class="fill" id="cb" style="width:0"></div></div>
<div class="row"><span id="cp"></span><span id="eta"></span></div></div>
<div class="card"><button id="pz" onclick="act('pause')">Pause</button><button id="rs" onclick="act('resume')">Resume</button>
<button id="sp" onclick="if(confirm('Stop for now? You can resume any time.'))act('stop')">Stop</button></div>
<div class="card"><pre id="lg"></pre></div></div>
<script>
function gb(b){return (b/1073741824).toFixed(2)}
function act(a){fetch('/'+a,{method:'POST'})}
async function tick(){try{const s=await (await fetch('/status')).json();
document.getElementById('ph').textContent=s.finished?'Finished. Safe to close.':(s.paused?'Paused (safe). Press Resume, or close and run shardpress again later.':s.phase);
const op=s.overall_total?100*s.overall_done/s.overall_total:0;
document.getElementById('ob').style.width=op.toFixed(1)+'%';
document.getElementById('ot').textContent=gb(s.overall_done)+' / '+gb(s.overall_total)+' GB ('+op.toFixed(1)+'%)';
document.getElementById('sd').textContent=s.done.length+' of '+s.range.length;
document.getElementById('fd').textContent=s.free_gb;
document.getElementById('cs').textContent=s.shard?('Shard '+s.shard+' of '+s.nshards):'Current shard';
const cp=s.bytes_total?100*s.bytes_now/s.bytes_total:0;
document.getElementById('cb').style.width=cp.toFixed(1)+'%';
document.getElementById('cp').textContent=s.pieces||(gb(s.bytes_now)+' / '+gb(s.bytes_total)+' GB');
document.getElementById('sp2').textContent=s.speed?(s.speed.toFixed(1)+' MB/s'):'';
const left=s.speed&&s.bytes_total?(s.bytes_total-s.bytes_now)/1048576/s.speed:0;
document.getElementById('eta').textContent=left?('~'+Math.ceil(left/60)+' min left'):'';
document.getElementById('lg').textContent=s.log.join('\\n');}catch(e){document.getElementById('ph').textContent='shardpress is not running (safe). Run it again to resume.'}}
setInterval(tick,1000);tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/status"):
            self._send(200, json.dumps(ST.snapshot()), "application/json")
        else:
            self._send(200, PAGE.replace("__REPO__", REPO.split("/")[-1]), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path == "/pause":
            ST.pause.set(); ST.interrupt(); ST.log("pause requested - stopping at a safe point")
        elif self.path == "/resume":
            ST.pause.clear(); ST.log("resumed")
        elif self.path == "/stop":
            ST.stop.set(); ST.pause.clear(); ST.interrupt(); ST.log("stop requested")
        self._send(200, "ok", "text/plain")


def start_server(open_browser):
    for port in range(8765, 8790):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/" % port
    print("status page: %s  (any browser; progress is also printed here)" % url, flush=True)
    if open_browser:
        try:
            webbrowser.open(url)                      # the system's default browser, whatever it is
        except Exception:
            pass
    return srv


def notify(msg):
    if sys.platform == "darwin":
        subprocess.call(["osascript", "-e", 'display notification "%s" with title "HoloMoE"' % msg.replace('"', "'")],
                        stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------ state

def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"repo": REPO, "sizes": {}, "converted": []}


def save_state(s):
    tmp = STATE + ".part"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE)


# ------------------------------------------------------------------ network (curl: resumes mid-file)

def base_url():
    return os.environ.get("HOLOMOE_BASE", "https://huggingface.co/%s/resolve/main/" % REPO)


def remote_size(name):
    out = subprocess.run(["curl", "-sIL", "--max-time", "60", base_url() + name], capture_output=True, text=True).stdout
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", out)
    return int(sizes[-1]) if sizes else 0


def wait_if_paused():
    while ST.pause.is_set() and not ST.stop.is_set():
        ST.set(phase="paused")
        time.sleep(0.5)


def download(name, dst, expected):
    """Download with resume. Returns True when dst is complete."""
    part = dst + ".part"
    while not ST.stop.is_set():
        wait_if_paused()
        if ST.stop.is_set():
            return False
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if expected and have == expected:
            os.replace(part, dst)
            return True
        if expected and have > expected:
            os.remove(part)
            continue
        need = (expected - have) + MARGIN
        if shutil.disk_usage(ROOT).free < need:
            ST.log("not enough free disk for %s (need %.1f GB). Free some space, then press Resume." % (name, need / 1024 ** 3))
            ST.pause.set()
            continue
        ST.set(phase="downloading " + name, bytes_total=expected, pieces="")
        ST.proc = subprocess.Popen(["curl", "-sL", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "3",
                                    "-o", part, base_url() + name])
        last_t, last_b = time.time(), have
        last_print = 0.0
        while ST.proc.poll() is None:
            time.sleep(0.5)
            b = os.path.getsize(part) if os.path.exists(part) else 0
            t = time.time()
            if t - last_t >= 2:
                ST.set(speed=(b - last_b) / (t - last_t) / 1048576)
                last_t, last_b = t, b
            ST.set(bytes_now=b, overall_done=ST.d["overall_base"] + b)
            if t - last_print >= 10 and expected:          # plain progress line for the terminal
                last_print = t
                sp = ST.d.get("speed") or 0
                eta = (expected - b) / 1048576 / sp if sp > 0 else 0
                print("  %s  %.2f / %.2f GB  %3.0f%%  %.1f MB/s%s" % (name, b / 1024 ** 3, expected / 1024 ** 3, 100.0 * b / expected, sp,
                      "  ~%d min left" % (eta // 60 + 1) if eta else ""), flush=True)
        rc = ST.proc.returncode
        ST.proc = None
        if ST.pause.is_set() or ST.stop.is_set():
            ST.log("download paused safely at %.2f GB" % (os.path.getsize(part) / 1024 ** 3 if os.path.exists(part) else 0))
            continue
        if rc != 0:
            ST.log("download interrupted (curl %d) - retrying in 10 s" % rc)
            time.sleep(10)
            continue
    return False


def check_shard(path):
    """A downloaded shard is complete when its own header says so."""
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hl))
    end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
    return os.path.getsize(path) == 8 + hl + end


def convert(path, H, I, NE):
    """Run holopress; returns True when every piece is written and verified."""
    while not ST.stop.is_set():
        wait_if_paused()
        if ST.stop.is_set():
            return False
        ST.set(phase="converting " + os.path.basename(path), speed=0)
        ST.proc = subprocess.Popen([HOLOPRESS, path, PLATE, str(H), str(I), str(NE)], stdout=subprocess.PIPE, text=True)
        ok = False
        last_pct = -10
        for line in ST.proc.stdout:
            line = line.strip()
            m = re.match(r"PIECE (\d+)/(\d+) (\w+) (.*)", line)
            if line.startswith("LAYER"):
                ST.log(line[6:])
            if m and m.group(3) == "DONE" and TEST_CONV_PAUSE.is_set():
                TEST_CONV_PAUSE.clear()
                ST.pause.set(); ST.interrupt(); ST.log("TEST: forced pause mid-conversion")
                threading.Timer(3.0, ST.pause.clear).start()
            if m:
                k, n = int(m.group(1)), int(m.group(2))
                ST.set(pieces="piece %d of %d  (%s)" % (k, n, m.group(4)[-48:]), bytes_now=k, bytes_total=n)
                pct = 100 * k // max(n, 1)
                if pct >= last_pct + 10 or k == n:
                    last_pct = pct
                    print("  converting %s: piece %d of %d (%d%%)" % (os.path.basename(path), k, n, pct), flush=True)
            elif line.startswith("PROGRESS"):
                ST.set(pieces=line[9:][-60:])
            elif line.startswith("DONE"):
                ok = True
        ST.proc.wait()
        rc = ST.proc.returncode
        ST.proc = None
        if ok and rc == 0:
            return True
        if ST.pause.is_set() or ST.stop.is_set():
            ST.log("conversion paused safely (finished pieces are kept)")
            continue
        ST.log("conversion failed (code %s) - see the Terminal for details" % rc)
        ST.stop.set()
        return False
    return False


# ------------------------------------------------------------------ main job

def parse_range(arg):
    NSHARDS = nshards()
    if arg in ("all", ""):
        return list(range(1, NSHARDS + 1))
    if arg == "test":
        return [TEST_SHARD or NSHARDS]
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", arg)
    if not m:
        sys.exit("range must look like 1-7, 3, all or test")
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    if not (1 <= a <= b <= NSHARDS):
        sys.exit("shards are numbered 1..%d" % NSHARDS)
    return list(range(a, b + 1))


def fetch_meta():
    os.makedirs(META, exist_ok=True)
    for name in META_FILES:
        dst = os.path.join(META, name)
        if os.path.exists(dst):
            continue
        tmp = dst + ".part"
        if subprocess.call(["curl", "-sL", "--fail", "--retry", "5", "-o", tmp, base_url() + name]) == 0:
            os.replace(tmp, dst)
        elif name in ("config.json", "model.safetensors.index.json"):
            sys.exit("could not download %s - check your internet connection" % name)


def model_dims():
    """hidden size, expert (or dense MLP) width, number of experts (0 = dense)"""
    with open(os.path.join(META, "config.json")) as f:
        c = json.load(f)
    t = c.get("text_config", c)
    ne = int(t.get("num_experts", 0) or 0)
    return int(t["hidden_size"]), int(t["moe_intermediate_size"] if ne else t["intermediate_size"]), ne


def load_shards():
    """shard file names, in order, from the model's own index (works for any shard count or naming)"""
    with open(os.path.join(META, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    SHARDS[:] = sorted(set(wm.values()))


def write_engine_meta():
    r = subprocess.run([sys.executable, os.path.join(HERE, "holomoe_meta.py"), PLATE], capture_output=True, text=True)
    for line in (r.stdout + r.stderr).strip().splitlines():
        ST.log(line)
    return r.returncode == 0


def run(shards, test_mode, open_browser):
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(PLATE, exist_ok=True)
    if not os.path.exists(HOLOPRESS):
        sys.exit("holopress is not built yet: run  make  in %s" % HERE)
    st = load_state()
    start_server(open_browser)
    ST.log("checking model files...")
    fetch_meta()
    load_shards()
    ST.set(nshards=nshards())
    if shards is None:
        shards = parse_range(ARG)
    H, I, NE = model_dims()
    for i in shards:
        n = shard_name(i)
        if n not in st["sizes"]:
            st["sizes"][n] = remote_size(n)
    save_state(st)
    todo = [i for i in shards if shard_name(i) not in st["converted"]]
    total = sum(st["sizes"][shard_name(i)] for i in shards)
    done_bytes = sum(st["sizes"][shard_name(i)] for i in shards if shard_name(i) in st["converted"])
    ST.set(range=shards, done=[i for i in shards if shard_name(i) in st["converted"]], overall_total=total,
           overall_done=done_bytes, overall_base=done_bytes)
    ST.log("%d of %d shards in this session already done" % (len(shards) - len(todo), len(shards)))

    for i in todo:
        if ST.stop.is_set():
            break
        n = shard_name(i)
        raw = os.path.join(RAW, n)
        exp = st["sizes"][n]
        ST.set(shard=i, bytes_now=0, bytes_total=exp, overall_base=done_bytes)
        if not (os.path.exists(raw) and check_shard(raw)):
            ST.log("shard %d: downloading %.2f GB" % (i, exp / 1024 ** 3))
            if test_mode and not os.path.exists(raw + ".part"):
                # self-test: stop partway through on purpose, then resume
                threading.Timer(8.0, lambda: (ST.pause.set(), ST.interrupt(), ST.log("TEST: forced pause mid-download"),
                                              threading.Timer(3.0, ST.pause.clear).start())).start()
            if not download(n, raw, exp):
                break
            if not check_shard(raw):
                ST.log("shard %d failed its integrity check - deleting and retrying next run" % i)
                os.remove(raw)
                ST.stop.set()
                break
            ST.log("shard %d: download verified" % i)
        if test_mode:
            TEST_CONV_PAUSE.set()
        ST.log("shard %d: converting to plate" % i)
        if not convert(raw, H, I, NE):
            break
        st["converted"].append(n)
        save_state(st)
        os.remove(raw)
        done_bytes += exp
        ST.set(done=ST.d["done"] + [i], overall_done=done_bytes, overall_base=done_bytes)
        ST.log("shard %d: converted, verified, raw file deleted" % i)
        notify("Shard %d of %d done" % (i, nshards()))

    left = [i for i in shards if shard_name(i) not in st["converted"]]
    if len(st["converted"]) == nshards():
        ST.log("all shards done - writing the engine's model.cfg")
        if write_engine_meta():
            flag = "" if MODEL == "35b" else " --model " + MODEL
            ST.log("ready: ./holomoe" + flag)
    if not left:
        ST.set(phase="finished", finished=True)
        ST.log("PASS - self-test complete: download, pause, resume, convert, verify all worked" if test_mode
               else "session complete")
        notify("Self-test passed" if test_mode else "Session complete")
    else:
        ST.set(phase="stopped safely - run the same command to resume")
        ST.log("stopped safely. %d shard(s) left in this session; run the same command to resume." % len(left))
    time.sleep(3)


def status_only():
    st = load_state()
    done = st.get("converted", [])
    if not os.path.exists(os.path.join(META, "model.safetensors.index.json")):
        print("%s: not started yet" % REPO)
        return
    load_shards()
    print("%s: converted %d of %d shards" % (REPO, len(done), nshards()))
    for i in range(1, nshards() + 1):
        n = shard_name(i)
        raw = os.path.join(RAW, n)
        mark = "done" if n in done else ("partial download" if os.path.exists(raw + ".part") else
                                         ("downloaded, not converted" if os.path.exists(raw) else "-"))
        print("  %2d  %-34s %s" % (i, n, mark))
    plate = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(PLATE) for f in fs)
    print("plate size so far: %.2f GB | free disk: %.1f GB" % (plate / 1024 ** 3, shutil.disk_usage(ROOT).free / 1024 ** 3))


def main():
    global ARG
    args, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
        elif a == "--model":
            skip = True
        elif not a.startswith("--"):
            args.append(a)
    arg = ARG = args[0] if args else "all"
    os.makedirs(ROOT, exist_ok=True)
    if arg in ("models", "list"):
        print("models (python3 shardpress.py all --model NAME):")
        for k, (repo, pd, line) in MODELS.items():
            print("  %-4s %s%s" % (k, line, "   [default]" if k == "35b" else ""))
        return
    if arg == "status":
        return status_only()
    if arg != "test" and arg not in ("all", "") and not re.fullmatch(r"\d+(-\d+)?", arg):
        sys.exit("range must look like 1-7, 3, all or test")
    signal.signal(signal.SIGINT, lambda *a: (ST.stop.set(), ST.interrupt(), ST.log("Ctrl-C: stopping at a safe point")))
    signal.signal(signal.SIGTERM, lambda *a: (ST.stop.set(), ST.interrupt()))
    run(None, arg == "test", "--no-browser" not in sys.argv)


if __name__ == "__main__":
    main()
