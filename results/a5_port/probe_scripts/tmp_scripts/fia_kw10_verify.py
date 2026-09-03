"""kw-10 验证序列: C05 首验 → C06/C07/C10 → C11×5 → 回归 C04/C08/C12/C18。"""
import glob, importlib.machinery, importlib.util, math, os, sys
import torch
import torch_npu

sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
from case_builder import load_cases, parse_case, build_inputs

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
print("ext:", _c)
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c)
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l)
_e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_e)
MLA_SCALE = 0.0721687836487032

def run_meta(inp, c):
    B = inp["kvlen_dev"].shape[0]
    return torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        c["H"], c["kvh"], c["D"], c["D"], actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=c["R"], block_size=c["BS"], aic_core_num=28, aiv_core_num=56)

def mla_sink(inp, c):
    return torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=run_meta(inp, c), num_query_heads=c["H"],
        num_key_value_heads=c["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
        sparse_mode=c["sparse_mode"], block_size=c["BS"])

def mla_official(inp, c):
    kw = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
              num_query_heads=c["H"], num_key_value_heads=c["kvh"], input_layout="TND",
              sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
              block_table=inp["bt"], block_size=c["BS"],
              actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
              return_softmax_lse=False)
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)[0]

def check(nm, runs=2):
    cases = load_cases()
    c = parse_case(cases[nm])
    inp = build_inputs(c)
    o = mla_official(inp, c).float().cpu()
    torch.npu.synchronize()
    meres, outs = [], []
    for i in range(runs):
        s = mla_sink(inp, c)[0]
        torch.npu.synchronize()
        outs.append(s.cpu())
        meres.append(((s.float().cpu() - o).abs() / o.abs().clamp_min(1e-6)).max().item())
    det = all(torch.equal(outs[0], x) for x in outs[1:]) if runs > 1 else None
    m = "/".join(f"{x:.4g}" for x in meres)
    ok = max(meres) <= 0.0079
    print(f"{nm:<22} mere={m:<24} bit_eq={det}  {'OK' if ok else 'FAIL'}")
    del inp
    torch.npu.empty_cache()
    return ok

print("== 首验 C05 ==")
check("C05_batch4", runs=3)
print("== 目标组 ==")
for nm in ["C06_batch16", "C07_ctx8k", "C10_decode_min_g2"]:
    check(nm, runs=3)
print("== C11 抖动 ×5 ==")
check("C11_gate_longseq", runs=5)
print("== 已收敛回归 ==")
for nm in ["C04_sig_b_min", "C08_ctx12k", "C12_r12_kv128k", "C18_r96_kv128k"]:
    check(nm, runs=2)
