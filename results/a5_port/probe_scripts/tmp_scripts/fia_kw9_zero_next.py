"""kw-9: 零化下一 batch 池区 — 污染 V 的来源判别。
mid config [8192, 8256, 8192×6]: b1 脏 (0.0709)。
变体: b2 的 K/V 池块全零 → 若脏消失 → B 侧读到 b2 的 V(越界)。
      b1 自己尾块之后的池内容(尾池块的后半)零化 → 若脏消失 → 读的是尾池块 pad。
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

cc = dict(case_name="zn", scenario="kw9", B=8, G=8, H=16, kvh=1, D=512, R=64, BS=128,
          T=64, q_len=[8] * 8, kvlen=[8192, 8256] + [8192] * 6,
          pool=sum(math.ceil(x / 128) for x in [8192, 8256] + [8192] * 6) + 2,
          bt_w=math.ceil(8256 / 128), use_mask=True, mask_rows=2048, mask_cols=2048,
          sparse_mode=3, gate=0, seed=20260901)
# b1 池块起点 64, b1 占 64..128 (65块), b2 从 129 起
def run(tag, mod=None):
    inp = build_inputs(cc)
    if mod:
        mod(inp)
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

def zero_b2(inp):
    k = inp["kpool"].clone(); kr = inp["krpool"].clone()
    k[129:193] = 0; kr[129:193] = 0  # b2 64块
    inp["kpool"].copy_(k); inp["krpool"].copy_(kr)

def zero_tailpad(inp):
    # b1 尾池块 = 128; 只零其无效后半 (列 64..128) — K/V rope 全零
    k = inp["kpool"].clone(); kr = inp["krpool"].clone()
    k[128, 0, 64:, :] = 0; kr[128, 0, 64:, :] = 0
    inp["kpool"].copy_(k); inp["krpool"].copy_(kr)

def zero_b2_v_only(inp):
    k = inp["kpool"].clone()
    k[129:193] = 0  # 注意: value==key 同一张量, 无法只零 V — 跳过
    inp["kpool"].copy_(k)

run("baseline_mid")
run("zero_tail_pool_pad", zero_tailpad)
run("zero_b2_pool", zero_b2)
