"""kw-9 Bug-A 边界扫描: B=1 G=8 H=16, kv 16k..128k → splits(uCore) vs mere 跳变。
buffer 容量: 6144B / (8rows*8pad*4B=256B) = 24 splits。预测: splits<=24 pass, >=25 崩。
同时 C08 x5 跑测 Bug-B 是否也影响'通过'case。
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

def run_meta(inp, c):
    N1, N2, D, R, BS = c["H"], c["kvh"], c["D"], c["R"], c["BS"]
    B = inp["kvlen_dev"].shape[0]
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)
    return meta

def mla_sink(inp, c):
    meta = run_meta(inp, c)
    return torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=meta, num_query_heads=c["H"],
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

def mk_case(tag, B, G, kv, seed, H=16, sparse=3, use_mask=True):
    return dict(case_name=tag, scenario="kw9", B=B, G=G, H=H, kvh=1, D=512, R=64, BS=128,
                T=B * G, q_len=[G] * B, kvlen=[kv] * B, pool=math.ceil(kv / 128) + 2,
                bt_w=math.ceil(kv / 128), use_mask=use_mask, mask_rows=2048, mask_cols=2048,
                sparse_mode=sparse, gate=0, seed=seed)

print(f"{'kv':>8}{'splits':>7}{'uCore':>7}{'uVecFd':>7}{'mere':>10}{'frac>0.02':>10}")
for kv in [16384, 32768, 49152, 57344, 65536, 81920, 98304, 114688, 131072]:
    cc = mk_case(f"bd{kv}", 1, 8, kv, 20260901)
    inp = build_inputs(cc)
    meta = run_meta(inp, cc).cpu().numpy()
    base = meta[576:586]
    s2spl = meta[0 * 10 + 6]
    s = mla_sink(inp, cc)[0].float().cpu()
    torch.npu.synchronize()
    o = mla_official(inp, cc).float().cpu()
    torch.npu.synchronize()
    rel = ((s - o).abs() / o.abs().clamp_min(1e-6))
    print(f"{kv:>8}{s2spl:>7}{base[4]:>7}{base[5]:>7}{rel.max():>10.4g}{(rel>0.02).float().mean():>10.5f}")
    del inp
    torch.npu.empty_cache()

# C08 x5 Bug-B 检查
cases = load_cases()
c08 = parse_case(cases["C08_ctx12k"])
inp = build_inputs(c08)
o = mla_official(inp, c08).float().cpu()
for i in range(5):
    s = mla_sink(inp, c08)[0].float().cpu()
    torch.npu.synchronize()
    rel = ((s - o).abs() / o.abs().clamp_min(1e-6))
    print(f"C08 run{i}: mere={rel.max():.4g} frac>0.02={(rel>0.02).float().mean():.5f}")
