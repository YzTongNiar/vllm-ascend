"""kw-9: C07 单脏源组合矩阵 — dirty 跟 kvlen? 跟 batch 序? 跟组合?"""
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
    return torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)

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

def mk(tag, kvs, seed=20260901):
    B = len(kvs)
    return dict(case_name=tag, scenario="kw9", B=B, G=8, H=16, kvh=1, D=512, R=64, BS=128,
                T=B * 8, q_len=[8] * B, kvlen=list(kvs),
                pool=sum(math.ceil(x / 128) for x in kvs) + 2,
                bt_w=max(math.ceil(x / 128) for x in kvs), use_mask=True, mask_rows=2048, mask_cols=2048,
                sparse_mode=3, gate=0, seed=seed)

def check(tag, cc, runs=2):
    inp = build_inputs(cc)
    B = cc["B"]
    m = run_meta(inp, cc).cpu().numpy()
    o = mla_official(inp, cc).float().cpu()
    torch.npu.synchronize()
    res = []
    for i in range(runs):
        s = mla_sink(inp, cc)[0].float().cpu()
        torch.npu.synchronize()
        rel = (s - o).abs() / o.abs().clamp_min(1e-6)
        res.append(rel.max().item())
    ratio = (mla_sink(inp, cc)[0].float().cpu() / o.clamp_min(1e-6)).view(B, 8, 16, -1)
    dirt_b = ((ratio > 1.02).any(-1)).sum(dim=(1, 2)).tolist()
    spl = [m[i * 10 + 6] for i in range(28)]
    print(f"[{tag}] uCore={m[580]} spl={spl[:B] if B <= 28 else ''} mere={'/'.join(f'{x:.4g}' for x in res)} "
          f"dirty_rows_per_batch={dirt_b}")
    del inp
    torch.npu.empty_cache()

check("B8_all8192", mk("a", [8192] * 8))
check("B8_onlyB1_8256", mk("b", [8192, 8256] + [8192] * 6))
check("B8_onlyB2_8256", mk("c", [8192, 8192, 8256] + [8192] * 5))
check("B8_onlyB1_8320", mk("d", [8192, 8320] + [8192] * 6))
check("B8_onlyB1_8208", mk("e", [8192, 8208] + [8192] * 6))
check("B2_8192_8256", mk("f", [8192, 8256]))
check("B4_8192x3_8256", mk("g", [8192, 8192, 8192, 8256]))
check("C07_orig", mk("h", [8192, 8256, 8320, 8384, 8448, 8512, 8576, 8640]))
