"""kw-9: C12 值代数判别 + C05 误差分布对照。"""
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
    return torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
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
    s = mla_sink(inp, c)[0].float().cpu()
    torch.npu.synchronize()
    o, lse = mla_official(inp, c)
    o = o.float().cpu(); lse = lse.float().cpu()
    torch.npu.synchronize()
    d = (s - o).abs()
    rel = (d / o.abs().clamp_min(1e-6))
    print(f"== {name}: mere={rel.max():.4g} max_abs={d.max():.4g} mean_abs={d.mean():.4g} "
          f"frac>0.02={ (rel>0.02).float().mean():.4g}")
    print("  sink[0,0,:6] :", [f"{x:.4f}" for x in s[0, 0, :6].tolist()])
    print("  off [0,0,:6] :", [f"{x:.4f}" for x in o[0, 0, :6].tolist()])
    print("  sink mean/std:", f"{s.mean():.4f}/{s.std():.4f}", " off mean/std:", f"{o.mean():.4f}/{o.std():.4f}")
    # 比值/差值结构
    ratio = (s / o.clamp_min(1e-6))
    print("  s/o ratio mean/std:", f"{ratio.mean():.4f}/{ratio.std():.4f}", " min/max:",
          f"{ratio.min():.3f}/{ratio.max():.3f}")
    print("  official lse[0,:4,0]:", [f"{x:.3f}" for x in lse.reshape(-1)[:8].tolist()])
    del inp, s, o
    torch.npu.empty_cache()
