"""kw-9: C05/C06/C07/C10 错误元素分布 — 按batch/head/token/gS1group 定位。"""
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

def mla_sink(inp, c):
    N1, N2, D, R, BS = c["H"], c["kvh"], c["D"], c["R"], c["BS"]
    B = inp["kvlen_dev"].shape[0]
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)
    return torch.ops._C_ascend.npu_fused_infer_attention_attention_dummy if False else \
        torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=meta, num_query_heads=N1,
        num_key_value_heads=N2, softmax_scale=MLA_SCALE, input_layout="TND",
        sparse_mode=c["sparse_mode"], block_size=BS)

def mla_official(inp, c):
    kw = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
              num_query_heads=c["H"], num_key_value_heads=c["kvh"], input_layout="TND",
              sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
              block_table=inp["bt"], block_size=c["BS"],
              actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
              return_softmax_lse=True)
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)

cases = load_cases()
for name in sys.argv[1:]:
    c = parse_case(cases[name])
    inp = build_inputs(c)
    B, G, H = c["B"], c["G"], c["H"]
    s = mla_sink(inp, c)[0].float().cpu()
    torch.npu.synchronize()
    o, lse = mla_official(inp, c)
    o = o.float().cpu()
    torch.npu.synchronize()
    d = (s - o).abs()
    rel = d / o.abs().clamp_min(1e-6)
    bad = rel > 0.02
    print(f"== {name} B={B} G={G} H={H}: mere={rel.max():.4g} frac_bad={bad.float().mean():.5f}")
    # TND out [T, H, D]; token t belongs to batch t//G
    badT = bad.view(B, G, H, -1)
    per_b = badT.float().mean(dim=(1, 2, 3))
    print("  per-batch frac_bad:", " ".join(f"{x:.4f}" for x in per_b.tolist()))
    per_g = badT.float().mean(dim=(0, 2, 3))
    print("  per-token-in-req frac_bad:", " ".join(f"{x:.4f}" for x in per_g.tolist()))
    per_h = badT.float().mean(dim=(0, 1, 3))
    print("  per-head frac_bad:", " ".join(f"{x:.3f}" for x in per_h.tolist()))
    # 错误元素的 ratio 分布
    rb = (s / o.clamp_min(1e-6))[bad]
    if rb.numel():
        print(f"  bad ratio: min={rb.min():.3f} med={rb.median():.3f} max={rb.max():.3f}")
    # gS1 row = token*H + head → FD group = row//8
    rows = bad.view(B, G, H, -1).any(dim=-1)  # [B,G,H] 该(token,head)是否有错
    rowbad = rows.permute(0, 2, 1).reshape(B, H * G)  # [B, row]  row = head*G+? 小心顺序
    # TND row序: token-major → row = g*H + h (g=token in req, h=head)
    rows2 = bad.view(B, G, H, -1).any(dim=-1).permute(0, 2, 1).reshape(B, H, G)
    print("  bad rows detail (batch0, head x token):")
    print("   ", "\n    ".join(" ".join("X" if rows2[0, h, g] else "." for g in range(G)) for h in range(min(H, 16))))
    del inp, s, o
    torch.npu.empty_cache()
