"""kw-9: stale-P(上一块)判别 — 把 b1 倒数第二块 K/KR 置 -100, 其 P 权重→0。
若膨胀源=上一块的过期 P → mere 塌缩。
"""
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

cc = dict(case_name="pb", scenario="kw9", B=8, G=8, H=16, kvh=1, D=512, R=64, BS=128,
          T=64, q_len=[8] * 8, kvlen=[8192, 8256] + [8192] * 6,
          pool=sum(math.ceil(x / 128) for x in [8192, 8256] + [8192] * 6) + 2,
          bt_w=math.ceil(8256 / 128), use_mask=True, mask_rows=2048, mask_cols=2048,
          sparse_mode=3, gate=0, seed=20260901)

def run(tag, mod_fn=None):
    inp = build_inputs(cc)
    if mod_fn:
        mod_fn(inp)
    o = mla_official(inp, cc).float().cpu()
    torch.npu.synchronize()
    meres = []
    for i in range(2):
        s = mla_sink(inp, cc)[0].float().cpu()
        torch.npu.synchronize()
        meres.append(((s - o).abs() / o.abs().clamp_min(1e-6)).max().item())
    print(f"[{tag}] mere={'/'.join(f'{x:.4g}' for x in meres)}")
    del inp
    torch.npu.empty_cache()

def kill_prev_block(inp):
    # b1 从池块 64 起; 倒数第二块 = b1 的列 [8256-64-512, 8256-64) → 池块 [64+ (8192-512)/128, ...)
    # b1 列 c → 池块 64 + c//128; 倒数第二块列 [7680, 8192) → 池块 64+60..64+63 → 池块 124..127
    k = inp["kpool"].clone()
    kr = inp["krpool"].clone()
    for pb in range(124, 128):
        k[pb] = -100.0
        kr[pb] = -100.0
    inp["kpool"].copy_(k)
    inp["krpool"].copy_(kr)

def kill_tail_prev2(inp):
    # 对照: 杀更早的块 (b1 列 [6656,7168) → 池块 116..119) — 不应影响
    k = inp["kpool"].clone()
    kr = inp["krpool"].clone()
    for pb in range(116, 120):
        k[pb] = -100.0
        kr[pb] = -100.0
    inp["kpool"].copy_(k)
    inp["krpool"].copy_(kr)

run("baseline")
run("kill_prev_block(124-127)", kill_prev_block)
run("kill_earlier_block(116-119)", kill_tail_prev2)
