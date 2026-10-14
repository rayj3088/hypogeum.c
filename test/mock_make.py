#!/usr/bin/env python3
"""Build a tiny checkpoint with real tensor names, layout and config keys so the
HoloMoE pipeline can be checked end to end.

  mock_make.py OUT_DIR [moe35 | dense | qwen3moe]
    moe35     Qwen3.6-35B-A3B family (hybrid attention, fused experts, shared expert)
    dense     Qwen3.5-2B family (hybrid attention, dense MLP, tied embeddings)
    qwen3moe  Qwen3-30B-A3B family (plain attention, unfused experts split across shards)"""
import json, os, struct, sys
import numpy as np

out = sys.argv[1]
ARCH = sys.argv[2] if len(sys.argv) > 2 else "moe35"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)

H, L, NH, HD, NKV = 256, 4, 4, 64, 2
LKH, LKD, LVH, LVD, CK = 2, 32, 4, 32, 4
E, TOPK, I, SI = 16, 4, 128, 128
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]

# ---------------- tokenizer: 256 byte tokens + a few merges + specials
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
b2u = bytes_to_unicode()
vocab = {b2u[b]: b for b in range(256)}
merges = []
for a, b in [("h", "e"), ("l", "l"), ("he", "ll"), ("Ġ", "w"), ("o", "r"), ("Ġw", "or"), ("Ġwor", "l"), ("Ġworl", "d"), ("a", "s"), ("Ċ", "Ċ")]:
    merges.append(a + " " + b); vocab[a + b] = len(vocab)
nreg = len(vocab)
VOCAB = 288
added = [{"id": nreg + i, "content": s, "special": True} for i, s in enumerate(SPECIALS)]
pat = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
tok = {"added_tokens": added,
       "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
           {"type": "Split", "pattern": {"Regex": pat}, "behavior": "Isolated", "invert": False},
           {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
       "model": {"type": "BPE", "vocab": vocab, "merges": merges}}
json.dump(tok, open(os.path.join(out, "tokenizer.json"), "w"), ensure_ascii=False)

if ARCH == "qwen3moe":
    json.dump({"chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"},
              open(os.path.join(out, "tokenizer_config.json"), "w"))
    cfg = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": H, "num_hidden_layers": L,
           "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD, "rope_theta": 10000000, "rms_norm_eps": 1e-6,
           "num_experts": E, "num_experts_per_tok": TOPK, "moe_intermediate_size": I, "intermediate_size": 384,
           "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [], "vocab_size": VOCAB, "tie_word_embeddings": False}
elif ARCH == "dense":
    DI = 384
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
       "text_config": {"model_type": "qwen3_5_text", "hidden_size": H, "num_hidden_layers": L, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                       "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD, "attn_output_gate": True,
                       "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "rope_type": "default"},
                       "rms_norm_eps": 1e-6, "intermediate_size": DI, "vocab_size": VOCAB,
                       "linear_num_key_heads": LKH, "linear_num_value_heads": LVH, "linear_key_head_dim": LKD,
                       "linear_value_head_dim": LVD, "linear_conv_kernel_dim": CK, "tie_word_embeddings": True}}
else:
  cfg = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
       "text_config": {"hidden_size": H, "num_hidden_layers": L, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                       "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
                       "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "rope_type": "default"},
                       "rms_norm_eps": 1e-6, "num_experts": E, "num_experts_per_tok": TOPK,
                       "moe_intermediate_size": I, "shared_expert_intermediate_size": SI, "vocab_size": VOCAB,
                       "linear_num_key_heads": LKH, "linear_num_value_heads": LVH, "linear_key_head_dim": LKD,
                       "linear_value_head_dim": LVD, "linear_conv_kernel_dim": CK, "tie_word_embeddings": False}}
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

# ---------------- weights
def bf16(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
T = {}
def W(name, shape, std, dtype="BF16", mean=0.0):
    a = (rng.standard_normal(shape) * std + mean).astype(np.float32)
    T[name] = (dtype, a)
PFX = "model." if ARCH == "qwen3moe" else "model.language_model."
NM = 1.0 if ARCH == "qwen3moe" else 0.0          # standard RMSNorm weights sit near 1, zero-centered near 0
W(PFX + "embed_tokens.weight", (VOCAB, H), 1.0)
if ARCH != "dense": W("lm_head.weight", (VOCAB, H), 0.06)
W(PFX + "norm.weight", (H,), 0.1, mean=NM)
kdim, vdim = LKH * LKD, LVH * LVD; cdim = 2 * kdim + vdim
for l in range(L):
    p = PFX + "layers.%d." % l
    W(p + "input_layernorm.weight", (H,), 0.1, mean=NM); W(p + "post_attention_layernorm.weight", (H,), 0.1, mean=NM)
    if l < 3 and ARCH != "qwen3moe":
        W(p + "linear_attn.in_proj_qkv.weight", (cdim, H), 0.06); W(p + "linear_attn.in_proj_z.weight", (vdim, H), 0.06)
        W(p + "linear_attn.in_proj_b.weight", (LVH, H), 0.06); W(p + "linear_attn.in_proj_a.weight", (LVH, H), 0.06)
        W(p + "linear_attn.out_proj.weight", (H, vdim), 0.06)
        W(p + "linear_attn.conv1d.weight", (cdim, 1, CK), 0.4)
        W(p + "linear_attn.A_log", (LVH,), 0.5, "F32"); W(p + "linear_attn.dt_bias", (LVH,), 0.5)
        W(p + "linear_attn.norm.weight", (LVD,), 0.1, mean=1.0)
    else:
        qm = 1 if ARCH == "qwen3moe" else 2
        W(p + "self_attn.q_proj.weight", (NH * HD * qm, H), 0.06); W(p + "self_attn.k_proj.weight", (NKV * HD, H), 0.06)
        W(p + "self_attn.v_proj.weight", (NKV * HD, H), 0.06); W(p + "self_attn.o_proj.weight", (H, NH * HD), 0.06)
        W(p + "self_attn.q_norm.weight", (HD,), 0.1, mean=NM); W(p + "self_attn.k_norm.weight", (HD,), 0.1, mean=NM)
    if ARCH == "dense":
        W(p + "mlp.gate_proj.weight", (DI, H), 0.06); W(p + "mlp.up_proj.weight", (DI, H), 0.06); W(p + "mlp.down_proj.weight", (H, DI), 0.06)
        continue
    W(p + "mlp.gate.weight", (E, H), 0.1)
    if ARCH == "qwen3moe":
        for e in range(E):
            W(p + "mlp.experts.%d.gate_proj.weight" % e, (I, H), 0.06); W(p + "mlp.experts.%d.up_proj.weight" % e, (I, H), 0.06)
            W(p + "mlp.experts.%d.down_proj.weight" % e, (H, I), 0.06)
        continue
    W(p + "mlp.experts.gate_up_proj", (E, 2 * I, H), 0.06); W(p + "mlp.experts.down_proj", (E, H, I), 0.06)
    W(p + "mlp.shared_expert.gate_proj.weight", (SI, H), 0.06); W(p + "mlp.shared_expert.up_proj.weight", (SI, H), 0.06)
    W(p + "mlp.shared_expert.down_proj.weight", (H, SI), 0.06); W(p + "mlp.shared_expert_gate.weight", (1, H), 0.06)
if ARCH != "qwen3moe": W("mtp.fc.weight", (H, 2 * H), 0.06)          # must be skipped

def write_shard(path, names):
    hdr, blobs, off = {}, [], 0
    for n in names:
        dt, a = T[n]
        raw = bf16(a).tobytes() if dt == "BF16" else a.astype(np.float32).tobytes()
        hdr[n] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw); off += len(raw)
    hdr["__metadata__"] = {"format": "pt"}
    h = json.dumps(hdr).encode(); h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for b in blobs: f.write(b)
names = list(T)
if ARCH == "qwen3moe":       # 3 shards, cut through the middle of layer 1's experts and layer 2's
    c1 = names.index("model.layers.1.mlp.experts.7.up_proj.weight"); c2 = names.index("model.layers.2.mlp.experts.11.down_proj.weight")
    parts = [names[:c1], names[c1:c2], names[c2:]]
else:
    half = len(names) // 2; parts = [names[:half], names[half:]]
index = {}
for i, part in enumerate(parts):
    fn = "model-%05d-of-%05d.safetensors" % (i + 1, len(parts))
    write_shard(os.path.join(out, fn), part)
    for n in part: index[n] = fn
json.dump({"metadata": {}, "weight_map": index}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
# the exact (bf16-rounded) weights for the numpy reference
np.savez(os.path.join(out, "ref_weights.npz"),
         **{n: (bf16(a).astype(np.uint32) << 16).view(np.float32) if dt == "BF16" else a for n, (dt, a) in T.items()})
print("mock written:", out, "vocab", VOCAB, "specials", [a["id"] for a in added])
