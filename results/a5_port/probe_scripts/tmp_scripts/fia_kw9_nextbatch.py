"""kw-9: 越界读假设判别 — 脏 batch 放池末尾(后续零) vs 中间(后续是下一batch数据)。"""
import glob, importlib.machinery, importlib.util, math, os, sys
import torch
import torch_npu

sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
from case_builder import build_inputs

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
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

def mk(tag, kvs, pad=0, seed=20260901):
    B = len(kvs)
    return dict(case_name=tag, scenario="kw9", B=B, G=8, H=16, kvh=1, D=512, R=64, BS=128,
                T=B * 8, q_len=[8] * B, kvlen=list(kvs),
                pool=sum(math.ceil(x / 128) for x in kvs) + 2 + pad,
                bt_w=max(math.ceil(x / 128) for x in kvs), use_mask=True, mask_rows=2048, mask_cols=2048,
                sparse_mode=3, gate=0, seed=seed)

def check(tag, cc, batch_idx, runs=2):
    inp = build_inputs(cc)
    B = cc["B"]
    o = mla_official(inp, cc).float().cpu()
    torch.npu.synchronize()
    meres = []
    for i in range(runs):
        s = mla_sink(inp, cc)[0].float().cpu()
        torch.npu.synchronize()
        meres.append(((s - o).abs() / o.abs().clamp_min(1e-6)).max().item())
    ratio = (mla_sink(inp, cc)[0].float().cpu() / o.clamp_min(1e-6)).view(B, 8, 16, -1)
    dirt_b = ((ratio > 1.02).any(-1)).sum(dim=(1, 2)).tolist()
    print(f"[{tag}] mere={'/'.join(f'{x:.4g}' for x in meres)} dirty/batch={dirt_b}")
    del inp
    torch.npu.empty_cache()

# 基线: b1=8256 在中间 (后续 b2..b7 数据紧随)
check("mid_8256@pos1", mk("a", [8192, 8256] + [8192] * 6), 1)
# 脏 batch 放最后 (池中其后全是 0)
check("last_8256@pos7", mk("b", [8192] * 7 + [8256]), 7)
# 脏 batch 放最后 + pool 大 padding
check("last_8256@pos7_pad", mk("c", [8192] * 7 + [8256], pad=8), 7)
