#!/usr/bin/env python3
"""
holomoe_meta.py - turn the model's config.json + tokenizer.json (in plate/meta)
into two simple files the C engine reads:  model.cfg  and  vocab.bin.

  python3 holomoe_meta.py [PLATE_DIR]        (default ~/holomoe/plate)

Standard library only.
"""
import json
import os
import struct
import sys

SUPPORTED = ["qwen3_5_moe", "qwen3_5", "qwen3_moe", "qwen3"]   # Qwen3.6/3.5 MoE, Qwen3.5 dense, Qwen3 MoE, Qwen3 dense
QWEN2_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def find(d, key):
    """first value for key anywhere inside nested dicts"""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def split_patterns(pre):
    out = []
    if not isinstance(pre, dict):
        return out
    if pre.get("type") == "Split":
        p = pre.get("pattern", {})
        out.append(p.get("Regex") or p.get("String") or "")
    for sub in pre.get("pretokenizers", []) or []:
        out += split_patterns(sub)
    return out


def main():
    plate = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/holomoe/plate")
    meta = os.path.join(plate, "meta")
    cfg = json.load(open(os.path.join(meta, "config.json")))
    t = cfg.get("text_config", cfg)
    tok = json.load(open(os.path.join(meta, "tokenizer.json"), encoding="utf-8"))

    arch = cfg.get("model_type") or t.get("model_type") or "?"
    hybrid = arch.startswith("qwen3_5") or arch.startswith("qwen3_next") or "linear_num_key_heads" in t
    if arch not in SUPPORTED:
        sys.exit("holomoe_meta: model type %r is not supported yet (supported: %s)" % (arch, ", ".join(SUPPORTED)))
    if t.get("mlp_only_layers") or int(t.get("decoder_sparse_step", 1)) != 1:
        sys.exit("holomoe_meta: mixed dense/MoE layer layouts are not supported yet")
    L = int(t["num_hidden_layers"])
    types = t.get("layer_types")
    if not hybrid:
        types = ["full_attention"] * L
    if not types:
        iv = int(t.get("full_attention_interval", 4))
        types = ["linear_attention" if (i + 1) % iv else "full_attention" for i in range(L)]
    lt = "".join("F" if x == "full_attention" else "L" for x in types[:L])
    head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
    rope = t.get("rope_parameters") or t.get("rope_scaling") or {}
    theta = float(rope.get("rope_theta", t.get("rope_theta", find(cfg, "rope_theta") or 10000.0)))
    prf = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", find(cfg, "partial_rotary_factor") or 1.0)))

    # tokenizer: vocab + merge ranks + specials
    dec = byte_decoder()
    vocab = tok["model"]["vocab"]
    merges = tok["model"].get("merges", [])
    rank = {}
    for i, m in enumerate(merges):
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        rank.setdefault(a + b, i)
    entries = {}
    for s, i in vocab.items():
        try:
            raw = bytes(dec[c] for c in s)
        except KeyError:
            raw = s.encode("utf-8")
        entries[i] = (raw, rank.get(s, -1 if len(raw) == 1 else len(merges) + i))
    specials = {}
    for a in tok.get("added_tokens", []):
        entries[a["id"]] = (a["content"].encode("utf-8"), -2)
        specials[a["content"]] = a["id"]
    ntok = max(entries) + 1

    pats = split_patterns(tok.get("pre_tokenizer"))
    pat = pats[0] if pats else ""
    digits = 3 if r"\p{N}{1,3}" in pat else 1
    norm = pat.replace(r"\p{N}{1,3}", r"\p{N}").replace(r"[\p{L}\p{M}]+", r"\p{L}+").replace(r"\p{L}\p{M}\p{N}", r"\p{L}\p{N}")
    pat_ok = norm == QWEN2_PATTERN

    def sid(name, fallback=-1):
        return specials.get(name, fallback)

    # does the chat template use <think>? (Qwen3-2507 Instruct does not)
    tmpl = ""
    for fn in ("chat_template.jinja", "tokenizer_config.json"):
        fp = os.path.join(meta, fn)
        if os.path.exists(fp):
            tmpl += open(fp, encoding="utf-8").read()
    think_ok = ("<think>" in tmpl) if tmpl else True
    n_exp = int(t.get("num_experts", 0) or 0)

    lines = {
        "dim": t["hidden_size"], "layers": L, "layer_types": lt,
        "n_heads": t["num_attention_heads"], "n_kv": t["num_key_value_heads"], "head_dim": head_dim,
        "rot_dim": int(head_dim * prf), "theta": theta, "eps": t.get("rms_norm_eps", 1e-6),
        "arch": arch, "norm_off": 1 if hybrid else 0, "attn_gate": int(bool(t.get("attn_output_gate", hybrid))),
        "n_exp": n_exp, "top_k": t.get("num_experts_per_tok", 0) if n_exp else 0,
        "moe_i": t.get("moe_intermediate_size", 0) if n_exp else 0, "shared_i": t.get("shared_expert_intermediate_size", 0) or 0,
        "dense_i": 0 if n_exp else t["intermediate_size"], "norm_topk": int(bool(t.get("norm_topk_prob", True))),
        "vocab": t["vocab_size"], "lin_kh": t.get("linear_num_key_heads", 16), "lin_vh": t.get("linear_num_value_heads", 32),
        "lin_kd": t.get("linear_key_head_dim", 128), "lin_vd": t.get("linear_value_head_dim", 128),
        "conv_k": t.get("linear_conv_kernel_dim", 4), "digits": digits,
        "im_start": sid("<|im_start|>"), "im_end": sid("<|im_end|>"), "eot": sid("<|endoftext|>"),
        "think": sid("<think>") if think_ok else -1, "think_end": sid("</think>") if think_ok else -1,
    }
    with open(os.path.join(meta, "model.cfg"), "w") as f:
        for k, v in lines.items():
            f.write("%s=%s\n" % (k, v))
    with open(os.path.join(meta, "vocab.bin"), "wb") as f:
        f.write(b"HMVB")
        f.write(struct.pack("<I", ntok))
        for i in range(ntok):
            raw, r = entries.get(i, (b"", -2))
            f.write(struct.pack("<Ii", len(raw), r))
            f.write(raw)
    shape = "%d experts top-%d" % (n_exp, lines["top_k"]) if n_exp else "dense MLP %d" % lines["dense_i"]
    print("model.cfg: %s, %d layers (%s...), dim %d, %s, vocab %d%s" % (arch, L, lt[:8], lines["dim"], shape, lines["vocab"], "" if think_ok else ", no thinking mode"))
    print("vocab.bin: %d tokens, %d merges, %d specials" % (ntok, len(merges), len(specials)))
    print("pre-tokenizer: %s (digits grouped by %d)" % ("Qwen2-style, supported" if pat_ok else "DIFFERENT - check the tokenizer:\n  " + pat, digits))


if __name__ == "__main__":
    main()
