#!/bin/sh
# End-to-end check of one model family on a tiny mock checkpoint:
#   mock -> holopress (every shard) -> holomoe_meta -> engine logits vs numpy reference
#   test/run_arch.sh WORKDIR [moe35|dense|qwen3moe]
set -e
W=$1; A=${2:-moe35}; T=$(cd "$(dirname "$0")" && pwd); R=$T/..
rm -rf "$W"; mkdir -p "$W"
python3 "$T/mock_make.py" "$W" "$A" >/dev/null
mkdir -p "$W/plate/meta"
cp "$W/config.json" "$W/tokenizer.json" "$W/plate/meta/"
[ -f "$W/tokenizer_config.json" ] && cp "$W/tokenizer_config.json" "$W/plate/meta/"
read H I NE <<EOF2
$(python3 -c "import json;c=json.load(open('$W/config.json'));t=c.get('text_config',c);print(t['hidden_size'],t.get('moe_intermediate_size',128),t.get('num_experts',0))")
EOF2
for s in "$W"/model-*.safetensors; do "$R/holopress" "$s" "$W/plate" "$H" "$I" "$NE" | grep -E "LAYER|DONE m" || true; done
python3 "$R/holomoe_meta.py" "$W/plate"
IDS="266,104,101,108,108,111,32,119,111,114,108,100,10,10,97,115,267,72,105"
for mode in exact int8; do
  flags="--full-vocab --no-learn"; [ $mode = exact ] && flags="$flags --bf16"
  "$R/holomoe" "$W/plate" $flags --dump "$IDS" "$W/eng_$mode.bin" 2>/dev/null
done
"$R/holomoe" "$W/plate" --full-vocab --no-learn --bf16 --batch --dump "$IDS" "$W/eng_batch.bin" 2>/dev/null
python3 "$T/mock_ref.py" "$W" "$IDS" "$W/ref.npy" ternary >/dev/null
python3 - "$W" <<'EOF2'
import sys, numpy as np
w = sys.argv[1]; ref = np.load(w + "/ref.npy"); n, V = ref.shape
def cmp(name, a):
    d = np.abs(a - ref[-len(a):]).max(); top = (a.argmax(-1) == ref[-len(a):].argmax(-1)).mean()
    return d, top
e = np.fromfile(w + "/eng_exact.bin", np.float32).reshape(n, V)
q = np.fromfile(w + "/eng_int8.bin", np.float32).reshape(n, V)
b = np.fromfile(w + "/eng_batch.bin", np.float32).reshape(1, V)
d, _ = cmp("exact", e); dq, tq = cmp("int8", q); db, _ = cmp("batch", b)
scale = np.abs(ref).max()
print("  exact path : max diff %.2e (logits up to %.1f)  %s" % (d, scale, "PASS" if d < 1e-4 * max(1, scale) else "FAIL"))
print("  batched    : max diff %.2e  %s" % (db, "PASS" if db < 1e-4 * max(1, scale) else "FAIL"))
print("  int8 path  : max diff %.2e, same top token %.0f%%" % (dq, 100 * tq))
EOF2
