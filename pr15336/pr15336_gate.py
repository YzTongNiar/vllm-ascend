"""PR #15336 设备门禁适配器：经 PR wrapper（_C_ascend 命名空间）驱动双算子，
vs 官方 npu_fused_infer_attention_score_v2 做 bit 对照。

用法（PYTHONPATH 指向 PR 树，使 vllm_ascend 解析到 PR 构建）:
  PYTHONPATH=/home/t00886357/fia_sink/pr15336-local python3 pr15336_gate.py mqa_c27
  python3 pr15336_gate.py C01
  python3 pr15336_gate.py kvlen_4096
  python3 pr15336_gate.py b2_248
"""
import importlib.util
import math
import os
import sys

BASE = __import__("os").environ.get("FIA_GATE_BASE", "/home/t00886357/fia_sink_a5_port_base")
PR_TREE = __import__("os").environ.get("FIA_GATE_TREE", BASE + "/vllm-ascend")
import os as _os
OURS = _os.environ.get("FIA_GATE_BASE", "/home/t00886357/fia_sink_a5_port_base")
sys.path.insert(0, os.path.join(OURS, "tests"))
sys.path.insert(0, PR_TREE)

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from gqa_case_builder import bits_equal, build_inputs, parse_case, resolve  # noqa: E402
from vllm_ascend.ops.fia_v2_sink import (  # noqa: E402
    build_fia_v2_sink_metadata,
    fused_infer_attention_score_v2_sink,
)


def call_pr_sink(inp, return_lse=False):
    """经 PR wrapper 调 sink（复制 wrapper 契约：两段式 + clone 语义在 wrapper 内）。"""
    c = inp["case"]
    meta = build_fia_v2_sink_metadata(
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        num_query_heads=c["n1"], num_key_value_heads=c["n2"],
        head_dim_qk=c["head_dim"], head_dim_v=c["head_dim"],
        input_layout=c["input_layout"], input_layout_kv="PA",
        sparse_mode=c["sparse_mode"], block_size=c["block_size"], rope_head_dim=0,
    )
    return fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["vpool"],
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], metadata=meta,
        num_query_heads=c["n1"], num_key_value_heads=c["n2"],
        softmax_scale=inp["scale"], input_layout=c["input_layout"],
        sparse_mode=c["sparse_mode"], block_size=c["block_size"],
    )


def gate_gqa(name):
    c = parse_case(resolve(name))
    inp = build_inputs(c)
    on = call_pr_sink(inp)
    ob = call_baseline(inp)
    eq, m, mx = bits_equal(on, ob)
    print(f"[PR_GATE {name}] bit_exact={eq} mere={m} max_abs={mx}")
    return eq


def gate_kvlen(kvlen):
    """D192 g1 B1 PA 长 kvlen（正确累计输入）。"""
    T, N1, N2, BS = 7, 8, 8, 128
    scale = 1.0 / math.sqrt(192)
    torch.manual_seed(21)
    q = (torch.rand(T, N1, 192) * 2).bfloat16().npu()
    pool = math.ceil(kvlen / BS) + 2
    kp = (torch.rand(pool, N2, BS, 192) * 2).bfloat16().npu()
    vp = (torch.rand(pool, N2, BS, 192) * 2).bfloat16().npu()
    bt = torch.arange(pool, dtype=torch.int32).view(1, -1).npu()
    ob, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q, kp, vp, num_query_heads=N1, num_key_value_heads=N2, input_layout="TND",
        sparse_mode=0, softmax_scale=scale, block_table=bt, block_size=BS,
        actual_seq_qlen=[T], actual_seq_kvlen=[kvlen], return_softmax_lse=False)
    meta = build_fia_v2_sink_metadata(
        actual_seq_qlen=torch.tensor([T], dtype=torch.int64, device="npu"),
        actual_seq_kvlen=torch.tensor([kvlen], dtype=torch.int64, device="npu"),
        num_query_heads=N1, num_key_value_heads=N2, head_dim_qk=192, head_dim_v=192,
        input_layout="TND", input_layout_kv="PA", sparse_mode=0, block_size=BS,
        rope_head_dim=0)
    on = fused_infer_attention_score_v2_sink(
        q, kp, vp, actual_seq_qlen=torch.tensor([T], dtype=torch.int64, device="npu"),
        actual_seq_kvlen=torch.tensor([kvlen], dtype=torch.int64, device="npu"),
        block_table=bt, metadata=meta, num_query_heads=N1, num_key_value_heads=N2,
        softmax_scale=scale, input_layout="TND", sparse_mode=0, block_size=BS)
    eq, m, mx = bits_equal(on, ob)
    print(f"[PR_GATE kvlen_{kvlen}] bit_exact={eq} mere={m} max_abs={mx}")
    return eq


def gate_b2(kvlen):
    qlen, B = 7, 2
    T = qlen * B
    N1, N2, D, BS = 8, 8, 192, 128
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(33)
    q = (torch.rand(T, N1, D) * 2).bfloat16().npu()
    nb = math.ceil(kvlen / BS)
    pool = nb * 2 + 2
    kp = (torch.rand(pool, N2, BS, D) * 2).bfloat16().npu()
    vp = (torch.rand(pool, N2, BS, D) * 2).bfloat16().npu()
    bt = torch.zeros(B, nb, dtype=torch.int32)
    bt[0, :] = torch.arange(nb)
    bt[1, :] = torch.arange(nb, 2 * nb)
    bt = bt.npu()
    refs = []
    for i in range(B):
        ob, _ = torch_npu.npu_fused_infer_attention_score_v2(
            q[i*qlen:(i+1)*qlen].contiguous(), kp, vp, num_query_heads=N1,
            num_key_value_heads=N2, input_layout="TND", sparse_mode=0,
            softmax_scale=scale, block_table=bt[i:i+1].contiguous(), block_size=BS,
            actual_seq_qlen=[qlen], actual_seq_kvlen=[kvlen], return_softmax_lse=False)
        refs.append(ob)
    ref = torch.cat(refs, 0)
    cum = torch.tensor([qlen, 2 * qlen], dtype=torch.int64, device="npu")
    klt = torch.tensor([kvlen, kvlen], dtype=torch.int64, device="npu")
    meta = build_fia_v2_sink_metadata(
        actual_seq_qlen=cum, actual_seq_kvlen=klt, num_query_heads=N1,
        num_key_value_heads=N2, head_dim_qk=D, head_dim_v=D, input_layout="TND",
        input_layout_kv="PA", sparse_mode=0, block_size=BS, rope_head_dim=0)
    on = fused_infer_attention_score_v2_sink(
        q, kp, vp, actual_seq_qlen=cum, actual_seq_kvlen=klt, block_table=bt,
        metadata=meta, num_query_heads=N1, num_key_value_heads=N2,
        softmax_scale=scale, input_layout="TND", sparse_mode=0, block_size=BS)
    eq, m, mx = bits_equal(on, ref)
    print(f"[PR_GATE b2_{kvlen}] bit_exact_vs_concat={eq} mere={m} max_abs={mx}")
    return eq


def call_baseline(inp):
    import torch_npu
    c = inp["case"]
    kw = dict(num_query_heads=c["n1"], num_key_value_heads=c["n2"],
              input_layout=c["input_layout"], sparse_mode=c["sparse_mode"],
              softmax_scale=inp["scale"], block_table=inp["bt"],
              block_size=c["block_size"], actual_seq_qlen=inp["qlen_host"],
              actual_seq_kvlen=inp["kvlen_host"], return_softmax_lse=False)
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        inp["q"], inp["kpool"], inp["vpool"], **kw)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "mqa_c27"
    if what.startswith("mqa_c2"):
        ok = gate_gqa(what)
    elif what.startswith("kvlen_"):
        ok = gate_kvlen(int(what.split("_")[1]))
    elif what.startswith("b2_"):
        ok = gate_b2(int(what.split("_")[1]))
    elif what.startswith("C2"):
        ok = gate_gqa(what)
    else:
        print(f"unknown gate {what}"); sys.exit(2)
    sys.exit(0 if ok else 1)
