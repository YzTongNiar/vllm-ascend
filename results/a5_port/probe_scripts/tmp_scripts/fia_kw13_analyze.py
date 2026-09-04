"""kw-13 分析: C10 batch0 core0 task 逐循环累计器快照 vs 宿主期望部分和比值对照。

期望构造 (AMLA 尺度无关): 对 token1 (odd-AIV rows) 各 head, S_L[c] = Σ_{j∈splitRange[0,endL)} p_j·V[j][c]
其中 p = 全 softmax 权重。快照比值 acc[L][r][ck]/acc[0][r][c0] 与期望比值 S_L[c]/S_0[c] 对照,
按循环定位 c2/c3 首次偏离。
"""
import glob, importlib.machinery, importlib.util, math, os, struct, sys
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

cases = load_cases()
c10 = parse_case(cases["C10_decode_min_g2"])
B, G, H, BS, D, R = c10["B"], c10["G"], c10["H"], c10["BS"], c10["D"], c10["R"]
inp = build_inputs(c10)
meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
    c10["H"], c10["kvh"], c10["D"], c10["D"], actual_seq_lengths=inp["qlen_dev"].clone(),
    actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
    sparse_mode=c10["sparse_mode"], input_layout="TND", input_layout_kv="PA",
    rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56).npu()
meta[604] = 0x600D600D
meta[605] = 0
torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"], key_rope=inp["krpool"],
    atten_mask=inp["mask"], actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
    block_table=inp["bt"], meta_data=meta, num_query_heads=c10["H"],
    num_key_value_heads=c10["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
    sparse_mode=c10["sparse_mode"], block_size=BS)
torch.npu.synchronize()
m = meta.cpu().numpy()

# core0 的任务范围 (s2End 表)
u = m[580]
prev_s2 = 0
ranges = []
for core in range(min(u, 6)):
    s2e = m[core * 10 + 3]
    ranges.append((core, prev_s2, s2e))
    prev_s2 = s2e
print("core s2 ranges:", [(c, a, b) for c, a, b in ranges[:4]])

# 期望部分和: token1 (j=1, 快照行 r → 任务行 16+r? — 注意快照是 core0 偶AIV? GetBlockIdx()==0 → 偶 AIV → 行 [0..16) = token0!
# 修正: 快照行 r ∈ [0,4) = 任务行 r = token0 的 head r (TND: row = token*H + head → token0 的 heads 0..3)
kv = c10["kvlen"][0]
ids = inp["bt"].cpu()[0, :math.ceil(kv / BS)].long()
klat = inp["kpool"].float().cpu()[ids].reshape(-1, D)[:kv]
krope = inp["krpool"].float().cpu()[ids].reshape(-1, R)[:kv]
q = inp["q"].float().cpu(); qpe = inp["qpe"].float().cpu()
j = 0  # token0
sc = (q[j] @ klat.T + qpe[j] @ krope.T) * MLA_SCALE
lim = kv - G + j + 1
sc[:, lim:] = float("-inf")
p = torch.softmax(sc, dim=-1)  # [H, kv]

snap = {}
for li in range(8):
    base = 700 + li * 24
    loop, bn2, splpos, bidx = int(m[base + 16]), int(m[base + 17]), int(m[base + 18]), int(m[base + 19])
    vals = [[struct.unpack("f", struct.pack("I", int(m[base + r * 4 + k])))[0] for k in range(4)] for r in range(4)]
    snap[li] = (loop, bn2, splpos, bidx, vals)
    nz = any(v != 0 for row in vals for v in row)
    print(f"loop{li}: loop={loop} bn2={bn2} split={splpos} b={bidx} {'DATA' if nz else '(empty)'}")
    for r in range(4):
        print(f"  r{r}(h{r}): {['%+.6e' % v for v in vals[r]]}")

# 期望: split0 的逐循环末 (core0 的 s2 范围 [0, s2End)) 每 512 一循环
core0_end = ranges[0][2] * 512  # s2 单位是块? 前面勘误: s2End 单位=512 列块
print(f"\ncore0 split range: [0, {core0_end}) cols, loops at 512 boundaries")
for li in sorted(snap.keys()):
    loop, bn2, splpos, bidx, vals = snap[li]
    if loop != li or bidx != 0 or splpos != 0:
        continue
    endL = min((li + 1) * 512, core0_end)
    if endL <= 0:
        continue
    part = p[:, :endL] @ klat[:endL]  # [H, 512] 期望部分和 (全权重归一)
    print(f"snapshot loop{li} (cols [0,{endL})):")
    for r in range(4):
        acc = vals[r]
        exp = [part[r, 64 + k * 128].item() for k in range(4)]
        # 归一化比较: 快照 c_k/c_0 vs 期望 e_k/e_0 (AMLA 尺度无关, 假定行内尺度一致)
        if acc[0] != 0 and exp[0] != 0:
            ar = [acc[k] / acc[0] for k in range(4)]
            er = [exp[k] / exp[0] for k in range(4)]
            print(f"  h{r}: acc_ratio={['%.4f' % x for x in ar]} exp_ratio={['%.4f' % x for x in er]}")
