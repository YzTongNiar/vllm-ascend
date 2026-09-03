"""kw-9: 指示V实验 — V[c]=1[c∈块B] → out_row = 该块 softmax 权重。
定位 b1(8256, tail=64) 脏行多算的到底是哪个块的贡献。无构建判别。
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

kv1 = 8256
cc = dict(case_name="ind", scenario="kw9", B=8, G=8, H=16, kvh=1, D=512, R=64, BS=128,
          T=64, q_len=[8] * 8, kvlen=[8192, kv1] + [8192] * 6,
          pool=sum(math.ceil(x / 128) for x in [8192, kv1] + [8192] * 6) + 2,
          bt_w=math.ceil(8640 / 128), use_mask=True, mask_rows=2048, mask_cols=2048,
          sparse_mode=3, gate=0, seed=20260901)
inp = build_inputs(cc)
B, G, H, BS = cc["B"], cc["G"], cc["H"], cc["BS"]

# 指示 V: 池化块编号 → 列区间。b1 连续占据池列 [start1, start1+kv1)
blk = [math.ceil(x / BS) for x in cc["kvlen"]]
start1 = blk[0]
def vind(block_lo, block_hi):
    """V=1 for b1 的全局列 [block_lo*512, block_hi*512)"""
    v = torch.zeros(cc["pool"], 1, BS, 512, dtype=torch.bfloat16)
    col0 = start1 * BS + block_lo * 512
    coln = start1 * BS + block_hi * 512
    for col in range(col0, min(coln, start1 * BS + kv1)):
        pb, off = col // BS, col % BS
        v[pb, 0, off, :] = 1.0
    return v.npu()

for (lo, hi, tag) in [(15, 16, "block15(last-full)"), (16, 17, "block16(tail64)")]:
    vpool = vind(lo, hi)
    kw_o = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
                num_query_heads=H, num_key_value_heads=1, input_layout="TND",
                sparse_mode=3, softmax_scale=MLA_SCALE, block_table=inp["bt"], block_size=BS,
                actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
                return_softmax_lse=False, atten_mask=inp["mask"])
    o = torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], vpool, **kw_o)[0].float().cpu()
    torch.npu.synchronize()
    s = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], vpool, query_rope=inp["qpe"], key_rope=inp["krpool"],
        atten_mask=inp["mask"], actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=run_meta(inp, cc), num_query_heads=H,
        num_key_value_heads=1, softmax_scale=MLA_SCALE, input_layout="TND",
        sparse_mode=3, block_size=BS)[0].float().cpu()
    torch.npu.synchronize()
    # b1 各行: out = 该块权重。脏行判定需普通V baseline — 直接给 b1 每行 off vs sink 权重
    ow = o.view(B, G, H, -1)[1].mean(-1)  # [G,H] 平均权重
    sw = s.view(B, G, H, -1)[1].mean(-1)
    dif = (sw - ow)
    print(f"== V-ind {tag}: b1 权重 sink-off 均值差 max={dif.max():.5f} min={dif.min():.5f} sum={dif.sum():.5f}")
    # 最差 3 行
    flat = dif.view(-1)
    idx = flat.abs().topk(3).indices.tolist()
    for i in idx:
        g, h = i // H, i % H
        print(f"   t{g} h{h}: off_w={ow[g,h]:.5f} sink_w={sw[g,h]:.5f} d={dif[g,h]:+.5f}")
    del vpool, o, s
    torch.npu.empty_cache()
