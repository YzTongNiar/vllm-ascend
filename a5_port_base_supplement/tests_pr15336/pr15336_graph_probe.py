"""PR #15336 图模式裁决探针（自包含可移植版；MLA K1/K2 判别形态）。
裁决三选一：DYNAMIC_OK（可用）/ FROZEN（不可用，F1 同款）/ MIXED（附数据）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PR_TREE = os.environ.get("FIA_GATE_TREE", os.path.join(PKG, "vllm-ascend"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PR_TREE, "python"))

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
K1, K2 = 2560, 5120
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

meta_static = build_meta(kvlen)
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
    print("[裁决] DYNAMIC_OK（更新生效 → PR 图模式可用）")
elif d1_1 == 0 and d1_2 > 0.01:
    print("[裁决] FROZEN（输出停在捕获值 → 不可用，F1 同款）")
else:
    print(f"[裁决] MIXED（部分生效，需附数据分析） d1_1={d1_1:.4f} d1_2={d1_2:.4f}")
