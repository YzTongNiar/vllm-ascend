"""kw-9b 构建后验证: Bug B (nupdate→V2读 竞态) 修复 + Bug A 保持。
"""
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
              return_softmax_lse=False)
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)[0]

def check(tag, cc, runs=1):
    inp = build_inputs(cc)
    o = mla_official(inp, cc).float().cpu()
    outs, meres = [], []
    for i in range(runs):
        s = mla_sink(inp, cc)[0]
        torch.npu.synchronize()
        sf = s.float().cpu()
        rel = ((sf - o).abs() / o.abs().clamp_min(1e-6))
        meres.append(rel.max().item())
        outs.append(sf)
    det = all(torch.equal(outs[0], x) for x in outs[1:]) if runs > 1 else None
    print(f"[{tag}] mere={'/'.join(f'{m:.4g}' for m in meres)} frac>0.02="
          f"{((outs[0]-o).abs()/o.abs().clamp_min(1e-6) > 0.02).float().mean():.5f} "
          f"rerun_bit_equal={det}")
    del inp
    torch.npu.empty_cache()
    return meres[-1]

cases = load_cases()
# Bug B 组 ×5
for nm in ["C05_batch4", "C06_batch16", "C07_ctx8k", "C10_decode_min_g2"]:
    check(nm, parse_case(cases[nm]), runs=5)
# Bug A 组保持
for nm in ["C04_sig_b_min", "C12_r12_kv128k", "C14_r12_kv512k", "C17_r24_kv512k"]:
    check(nm, parse_case(cases[nm]), runs=2)
# 收敛组保持
for nm in ["C08_ctx12k", "C11_gate_longseq", "C18_r96_kv128k"]:
    check(nm, parse_case(cases[nm]), runs=2)
