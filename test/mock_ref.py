#!/usr/bin/env python3
"""Independent numpy forward pass of Qwen3.5/3.6 (MoE and dense) and Qwen3-MoE,
written from the official transformers modeling code, used to check holomoe.c on
the mock models.

  mock_ref.py MOCK_DIR "ids" OUT.npy [ternary|float]
ternary: experts are read back from the plate's ternary files (what the engine uses)
float:   the original full-precision experts (shows what ternarizing costs)"""
import json, os, struct, sys
import numpy as np

d, ids, outp = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else "ternary"
Wt = np.load(os.path.join(d, "ref_weights.npz"))
C0 = json.load(open(os.path.join(d, "config.json"))); c = C0.get("text_config", C0)
ARCH = C0["model_type"]
Q3 = ARCH == "qwen3_moe"                       # Qwen3MoE: standard norm, plain attention, full rotary, unfused experts
DENSE = "num_experts" not in c
PFX = "model." if Q3 else "model.language_model."
if Q3:
    c["layer_types"] = ["full_attention"] * c["num_hidden_layers"]
    c["rope_parameters"] = {"rope_theta": c["rope_theta"], "partial_rotary_factor": 1.0}
    for k in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim"): c[k] = 1
H, NL = c["hidden_size"], c["num_hidden_layers"]
NH, NKV, HD = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
ROT = int(HD * c["rope_parameters"]["partial_rotary_factor"]); THETA = c["rope_parameters"]["rope_theta"]
EPS = c["rms_norm_eps"]; E, K = c.get("num_experts", 0), c.get("num_experts_per_tok", 0); I = c.get("moe_intermediate_size", 0)
LKH, LVH, LKD, LVD, CK = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"], c["linear_conv_kernel_dim"]
f64 = np.float64

def w(n): return Wt[n].astype(f64)
def rms(x, wt):  # Qwen3_5 RMSNorm: x * rsqrt(mean x^2 + eps) * (1 + w);  Qwen3MoeRMSNorm: ... * w
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * ((0.0 if Q3 else 1.0) + wt)
def silu(x): return x / (1 + np.exp(-x))
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softplus(x): return np.log1p(np.exp(x))

def read_tern(path):
    b = open(path, "rb").read()
    h = struct.unpack("<16I", b[:64]); Ee, nmat, rows, cols, gpr = h[2], h[3], h[4], h[5], h[7]
    mb = rows * gpr * 34; mats = np.zeros((Ee, nmat, rows, cols))
    for e in range(Ee):
        for m in range(nmat):
            o = 64 + (e * nmat + m) * mb
            mk = np.frombuffer(b, np.uint64, rows * gpr * 4, o).reshape(rows, gpr, 4)
            sc = np.frombuffer(b, np.uint16, rows * gpr, o + rows * gpr * 32).astype(np.uint32) << 16
            sc = sc.view(np.float32).reshape(rows, gpr).astype(f64)
            bits = lambda a, bb: ((mk[:, :, a:bb + 1, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).reshape(rows, gpr, 128).astype(f64)
            t = bits(0, 1) - bits(2, 3)
            mats[e, m] = (t * sc[:, :, None]).reshape(rows, cols)
    return mats

T = len(ids)
x = w(PFX + "embed_tokens.weight")[ids]
pos = np.arange(T)
inv = 1.0 / (THETA ** (np.arange(0, ROT, 2) / ROT))
ang = pos[:, None] * inv[None]; cos = np.cos(np.concatenate([ang, ang], -1)); sin = np.sin(np.concatenate([ang, ang], -1))
def rot_half(v): h = v.shape[-1] // 2; return np.concatenate([-v[..., h:], v[..., :h]], -1)
def rope(v):  # v [T, heads, HD]; partial rotary on first ROT dims
    r, p = v[..., :ROT], v[..., ROT:]
    return np.concatenate([r * cos[:, None] + rot_half(r) * sin[:, None], p], -1)

for l in range(NL):
    p = PFX + "layers.%d." % l
    hn = rms(x, w(p + "input_layernorm.weight"))
    if c["layer_types"][l] == "full_attention":
        if Q3:
            q = (hn @ w(p + "self_attn.q_proj.weight").T).reshape(T, NH, HD); gate = None
        else:
            qg = (hn @ w(p + "self_attn.q_proj.weight").T).reshape(T, NH, 2 * HD)
            q, gate = qg[..., :HD], qg[..., HD:].reshape(T, NH * HD)
        q = rms(q, w(p + "self_attn.q_norm.weight"))
        k = rms((hn @ w(p + "self_attn.k_proj.weight").T).reshape(T, NKV, HD), w(p + "self_attn.k_norm.weight"))
        v = (hn @ w(p + "self_attn.v_proj.weight").T).reshape(T, NKV, HD)
        q, k = rope(q), rope(k)
        k = np.repeat(k, NH // NKV, axis=1); v = np.repeat(v, NH // NKV, axis=1)
        s = np.einsum("thd,shd->hts", q, k) / np.sqrt(HD)
        s = np.where(np.tril(np.ones((T, T)))[None] > 0, s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", s, v).reshape(T, NH * HD)
        if gate is not None: o = o * sigmoid(gate)
        x = x + o @ w(p + "self_attn.o_proj.weight").T
    else:
        kd, vd = LKH * LKD, LVH * LVD
        qkv = hn @ w(p + "linear_attn.in_proj_qkv.weight").T          # [T, cdim]
        z = (hn @ w(p + "linear_attn.in_proj_z.weight").T).reshape(T, LVH, LVD)
        b = hn @ w(p + "linear_attn.in_proj_b.weight").T
        a = hn @ w(p + "linear_attn.in_proj_a.weight").T
        cw = w(p + "linear_attn.conv1d.weight")[:, 0, :]              # [cdim, CK]
        pad = np.concatenate([np.zeros((CK - 1, qkv.shape[1])), qkv], 0)
        conv = np.stack([(pad[t:t + CK] * cw.T).sum(0) for t in range(T)])
        conv = silu(conv)
        q = conv[:, :kd].reshape(T, LKH, LKD); k = conv[:, kd:2 * kd].reshape(T, LKH, LKD); v = conv[:, 2 * kd:].reshape(T, LVH, LVD)
        beta = sigmoid(b); g = -np.exp(w(p + "linear_attn.A_log")) * softplus(a + w(p + "linear_attn.dt_bias"))
        q = np.repeat(q, LVH // LKH, axis=1); k = np.repeat(k, LVH // LKH, axis=1)   # repeat_interleave
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6); k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        q = q / np.sqrt(LKD)
        S = np.zeros((LVH, LKD, LVD)); out = np.zeros((T, LVH, LVD))
        for t in range(T):
            S = S * np.exp(g[t])[:, None, None]
            kv = (S * k[t][:, :, None]).sum(1)
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            out[t] = (S * q[t][:, :, None]).sum(1)
        nw = w(p + "linear_attn.norm.weight")                          # gated norm: plain weight
        out = out / np.sqrt((out * out).mean(-1, keepdims=True) + EPS) * nw * silu(z)
        x = x + out.reshape(T, vd) @ w(p + "linear_attn.out_proj.weight").T
    hn = rms(x, w(p + "post_attention_layernorm.weight"))
    if DENSE:                                                          # Qwen3_5MLP
        x = x + (silu(hn @ w(p + "mlp.gate_proj.weight").T) * (hn @ w(p + "mlp.up_proj.weight").T)) @ w(p + "mlp.down_proj.weight").T
        continue
    probs = hn @ w(p + "mlp.gate.weight").T
    probs = np.exp(probs - probs.max(-1, keepdims=True)); probs /= probs.sum(-1, keepdims=True)
    if mode == "float" and Q3:
        GATE = np.stack([w(p + "mlp.experts.%d.gate_proj.weight" % e) for e in range(E)])
        UP = np.stack([w(p + "mlp.experts.%d.up_proj.weight" % e) for e in range(E)])
        DOWN = np.stack([w(p + "mlp.experts.%d.down_proj.weight" % e) for e in range(E)])
    elif mode == "float":
        gu = w(p + "mlp.experts.gate_up_proj"); dn = w(p + "mlp.experts.down_proj")
        GATE, UP, DOWN = gu[:, :I], gu[:, I:], dn
    else:
        a1 = read_tern(os.path.join(d, "plate/experts/L%02d.gu" % l)); a2 = read_tern(os.path.join(d, "plate/experts/L%02d.dn" % l))
        GATE, UP, DOWN = a1[:, 0], a1[:, 1], a2[:, 0]
    moe = np.zeros_like(x)
    for t in range(T):
        top = np.argsort(-probs[t], kind="stable")[:K]; wts = probs[t][top]
        if c.get("norm_topk_prob", True): wts = wts / wts.sum()
        for e, wt in zip(top, wts):
            moe[t] += wt * (DOWN[e] @ (silu(GATE[e] @ hn[t]) * (UP[e] @ hn[t])))
    if Q3:
        x = x + moe; continue
    sh = (silu(hn @ w(p + "mlp.shared_expert.gate_proj.weight").T) * (hn @ w(p + "mlp.shared_expert.up_proj.weight").T)) @ w(p + "mlp.shared_expert.down_proj.weight").T
    sh = sh * sigmoid(hn @ w(p + "mlp.shared_expert_gate.weight").T)
    x = x + moe + sh
head = w("lm_head.weight") if "lm_head.weight" in Wt.files else w(PFX + "embed_tokens.weight")   # tied embeddings
logits = rms(x, w(PFX + "norm.weight")) @ head.T
np.save(outp, logits.astype(np.float32))
print("reference logits", logits.shape, "mode", mode)
