"""PR #15336 图模式裁决探针（MLA 形态，判据与 aclgraph_decisive_replay_probe.py 同口径）。

PR 设计：metadata 由 builder 在图【外】eager 构建（AICPU 不入图，绕开 F1 的
"replay 不重执行 AICPU"），图内只捕获 FIA 主 kernel；重放前 builder 把新分核计划
写回同一静态 metadata buffer → device-side tiling 应使重放响应新 kvlen。

判据三选一：
  DYNAMIC_OK   重放(更新后) == eager(K2)  → PR 图模式可用
  FROZEN       重放(更新后) == eager(K1)  → 输出停在捕获值 → 不可用
  MIXED        两者都不是                  → 有条件（附数据）
"""
import math
import os
import sys

BASE = __import__("os").environ.get("FIA_GATE_BASE", "/home/t00886357/fia_sink_a5_port_base")
PR_TREE = __import__("os").environ.get("FIA_GATE_TREE", BASE + "/vllm-ascend")
sys.path.insert(0, os.path.join("/home/t00886357/npu_fused_infer_attention_score_v2_sink", "tests"))
sys.path.insert(0, PR_TREE)

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from vllm_ascend.ops.fia_v2_sink import (  # noqa: E402
    build_fia_v2_sink_metadata,
    fused_infer_attention_score_v2_sink,
)

DEV = "npu"
B, G, N1, N2 = 8, 8, 16, 1
D, DR, BS = 512, 64, 128
POOL = 1000
K1 = 2560
K2 = 5120
BT_W = 40
SCALE = 0.0721687836487032

torch.manual_seed(7)
q = (torch.rand(B * G, N1, D, dtype=torch.bfloat16) * 2).to(DEV)
qr = (torch.rand(B * G, N1, DR, dtype=torch.bfloat16) * 2).to(DEV)
kc = (torch.rand(POOL, N2, BS, D, dtype=torch.bfloat16) * 2).to(DEV)
kr = (torch.rand(POOL, N2, BS, DR, dtype=torch.bfloat16) * 2).to(DEV)
perm = torch.randperm(POOL)
bt = perm[: B * BT_W].reshape(B, BT_W).to(torch.int32).to(DEV)
qlen = torch.tensor([G * (i + 1) for i in range(B)], dtype=torch.int64, device=DEV)
kvlen = torch.tensor([K1] * B, dtype=torch.int64, device=DEV)
kvlen += torch.arange(B, device=DEV)


def build_meta(kv_dev):
    return build_fia_v2_sink_metadata(
        actual_seq_qlen=qlen, actual_seq_kvlen=kv_dev,
        num_query_heads=N1, num_key_value_heads=N2,
        head_dim_qk=D, head_dim_v=D, input_layout="TND", input_layout_kv="PA",
        sparse_mode=0, block_size=BS, rope_head_dim=DR)


def pr_eager(kv_vals):
    kv = torch.tensor(kv_vals, dtype=torch.int64, device=DEV)
    meta = build_meta(kv)
    o = fused_infer_attention_score_v2_sink(
        q, kc, kc, actual_seq_qlen=qlen, actual_seq_kvlen=kv, block_table=bt,
        metadata=meta, num_query_heads=N1, num_key_value_heads=N2,
        softmax_scale=SCALE, input_layout="TND", sparse_mode=0, block_size=BS,
        query_rope=qr, key_rope=kr)
    torch.npu.synchronize()
    return o.clone()


K1v = kvlen.cpu().tolist()
K2v = [v + K2 - K1 for v in K1v]
ref1 = pr_eager(K1v)
ref2 = pr_eager(K2v)
d_ref = (ref1.float() - ref2.float()).abs().max().item()
print(f"[sanity] ref1 vs ref2 max_abs = {d_ref:.4f}（应 >> 0）", flush=True)
assert d_ref > 0.01

# ---- PR 图路径捕获：metadata 图外建好，图内只有 FIA kernel ----
meta_static = build_meta(kvlen)          # 静态 metadata buffer（K1 计划）
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    out_cap = fused_infer_attention_score_v2_sink(
        q, kc, kc, actual_seq_qlen=qlen, actual_seq_kvlen=kvlen, block_table=bt,
        metadata=meta_static, num_query_heads=N1, num_key_value_heads=N2,
        softmax_scale=SCALE, input_layout="TND", sparse_mode=0, block_size=BS,
        query_rope=qr, key_rope=kr)

g.replay()
o0 = out_cap.clone()
torch.npu.synchronize()
d0_1 = (o0.float() - ref1.float()).abs().max().item()
print(f"[固定输入重放] vs eager(K1) max_abs = {d0_1:.6f} → {'一致' if d0_1 == 0 else '不一致'}", flush=True)

# ---- 重放前：builder 图外重建 metadata（K2 计划写回同一静态 buffer）----
kvlen.copy_(torch.tensor(K2v, dtype=torch.int64, device=DEV))
meta_new = build_meta(kvlen)
meta_static[:1024].copy_(meta_new[:1024])
g.replay()
o1 = out_cap.clone()
torch.npu.synchronize()
d1_2 = (o1.float() - ref2.float()).abs().max().item()
d1_1 = (o1.float() - ref1.float()).abs().max().item()
print(f"[更新metadata后重放] vs eager(K2) max_abs = {d1_2:.6f} | vs eager(K1) max_abs = {d1_1:.6f}", flush=True)
if d1_2 == 0 and d1_1 > 0.01:
    verdict = "DYNAMIC_OK（更新生效 → PR 图模式可用）"
elif d1_1 == 0 and d1_2 > 0.01:
    verdict = "FROZEN（输出停在捕获值 → 不可用，F1 同款）"
else:
    verdict = f"MIXED（部分生效，需附数据分析） d1_1={d1_1:.4f} d1_2={d1_2:.4f}"
print(f"[裁决] {verdict}", flush=True)
