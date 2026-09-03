"""kw-9a 构建后快检: Bug A 修复验证。
预期: C12/kv57344 (26-28 splits) mere 从 2.3-4.6 → ~1ulp 或 Bug-B 级小错;
C05/C10 竞态残留（本构建未修 Bug B）。
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

def mk_case(tag, B, G, kv, seed, H=16, sparse=3, use_mask=True):
    return dict(case_name=tag, scenario="kw9", B=B, G=G, H=H, kvh=1, D=512, R=64, BS=128,
                T=B * G, q_len=[G] * B, kvlen=[kv] * B, pool=math.ceil(kv / 128) + 2,
                bt_w=math.ceil(kv / 128), use_mask=use_mask, mask_rows=2048, mask_cols=2048,
                sparse_mode=sparse, gate=0, seed=seed)

def check(tag, cc, runs=1):
    inp = build_inputs(cc)
    o = mla_official(inp, cc).float().cpu()
    outs = []
    for i in range(runs):
        s = mla_sink(inp, cc)[0]
        torch.npu.synchronize()
        sf = s.float().cpu()
        rel = ((sf - o).abs() / o.abs().clamp_min(1e-6))
        print(f"[{tag} run{i}] mere={rel.max():.4g} frac>0.02={(rel > 0.02).float().mean():.5f}")
        outs.append(sf)
    if runs > 1:
        print(f"[{tag}] rerun bit-equal: {torch.equal(outs[0], outs[1])}")
    del inp
    torch.npu.empty_cache()

# Bug A 组: 长KV 头少 (26-28 splits)
check("C12_128k_s26", mk_case("c12", 1, 8, 131072, 20260901))
check("kv57344_s28", mk_case("k57", 1, 8, 57344, 20260901))
check("kv49152_s24", mk_case("k49", 1, 8, 49152, 20260901), runs=1)
# Bug B 组: 竞态残留预期
cases = load_cases()
for nm in ["C05_batch4", "C10_decode_min_g2"]:
    check(nm, parse_case(cases[nm]), runs=3)
