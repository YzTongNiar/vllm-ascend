"""kw-9: C12 误差模式判别 — 按头/行分解 + lse 对照。
区分: 单头全错(staging/合并寻址) vs diffuse(舍入累积) vs lse 也错(softmax链)。
"""
import glob, importlib.machinery, importlib.util, math, os, sys
import torch
import torch_npu

sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
from case_builder import load_cases, parse_case, build_inputs

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c)
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l)
_e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_e)

MLA_SCALE = 0.0721687836487032

def mla_sink(inp, c, want_lse=False):
    N1, N2, D, R, BS = c["H"], c["kvh"], c["D"], c["R"], c["BS"]
    B = inp["kvlen_dev"].shape[0]
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)
    outs = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=meta, num_query_heads=N1,
        num_key_value_heads=N2, softmax_scale=MLA_SCALE, input_layout="TND",
        sparse_mode=c["sparse_mode"], block_size=BS)
    return outs

def mla_official(inp, c, want_lse=True):
    kw = dict(
        query_rope=inp["qpe"], key_rope=inp["krpool"],
        num_query_heads=c["H"], num_key_value_heads=c["kvh"], input_layout="TND",
        sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
        block_table=inp["bt"], block_size=c["BS"],
        actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
        return_softmax_lse=want_lse)
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)

cases = load_cases()
name = sys.argv[1] if len(sys.argv) > 1 else "C12_r12_kv128k"
c = parse_case(cases[name])
inp = build_inputs(c)
B, G, H = c["B"], c["G"], c["H"]

s = mla_sink(inp, c)
torch.npu.synchronize()
o, o_lse = mla_official(inp, c)
torch.npu.synchronize()

print(f"case={name} B={B} G={G} H={H} sink_out={tuple(s[0].shape)} off_out={tuple(o.shape)}")
sf, of = s[0].float().cpu(), o.float().cpu()
d = (sf - of).abs()
rel = d / of.abs().clamp_min(1e-6)
print(f"mere(max-rel)={rel.max().item():.4g} max_abs={d.max().item():.4g} mean_abs={d.mean().item():.4g}")

# 输出布局 TND: [T, H, D]. 按头分解
per_head_max = d.amax(dim=(0, 2))
per_head_cnt = (d > 0.05).sum(dim=(0, 2))
print("per-head max_abs:", " ".join(f"{x:.3f}" for x in per_head_max.tolist()))
print("per-head #elem>0.05:", per_head_cnt.tolist())
# 按token分解
per_tok_max = d.amax(dim=(1, 2))
print("per-token max_abs:", " ".join(f"{x:.3f}" for x in per_tok_max.tolist()))

# lse 对照 (若有)
if o_lse is not None and len(o_lse) > 0:
    print("official lse shape:", tuple(o_lse.shape))
else:
    print("official lse: None")

# 误差分布: 多少元素超过 2ulp(0.0157)
big = (rel > 0.02).float().mean().item()
print(f"frac elem rel>0.02: {big:.4g}")
# sink 是否有非有限值
print(f"sink finite: {torch.isfinite(sf).all().item()}, official finite: {torch.isfinite(of).all().item()}")

# 确定性: 二跑
s2 = mla_sink(inp, c)
torch.npu.synchronize()
print("deterministic rerun bit-equal:", torch.equal(s2[0].cpu(), s[0].cpu()))
