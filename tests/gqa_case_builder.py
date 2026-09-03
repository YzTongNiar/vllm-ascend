"""GQA/MQA case builder + bit comparison — a5_port migration rebuild (2026-09-02).

原版（原工作区 tests/gqa_case_builder.py）未随交接包携带且原工作区已不存在；
本文件按 pr15336_gate.py / pr15336_gate_c01.py 的消费契约重建，覆盖：
  resolve(name) → parse_case(c) → build_inputs(c) → {q,kpool,vpool,bt,qlen_dev,
  kvlen_dev,qlen_host,kvlen_host,scale}；bits_equal(on, ob) → (eq, mere, max_abs)。

⚠️ case 参数为迁移重建值（按《FIA_v2_Sink_算子接口与实现说明.md》§6 支持矩阵选取
代表性形态），非原工作区原值；覆盖语义（头维组合/布局/变长）对齐门禁清单。
"""
import math

import torch

# ---------------------------------------------------------------- cases ----
# 字段：n1/n2 头数（g = n1/n2），head_dim (=D_qk=D_v，GQA 非 MLA)，
# input_layout（q 布局，TND 累计契约），sparse_mode，block_size（PA 页大小），
# T/q_len，kv_len，B（batch），seed。
CASES = {
    # MQA 极限（N2=1, gS=n1）× D192（round_4 (192,192,0) 族），短 kv
    "mqa_c27": dict(n1=8, n2=1, head_dim=192, input_layout="TND", sparse_mode=0,
                    block_size=128, q_len=7, kv_len=256, B=1, seed=27),
    # MQA × D192 长 kv（FD 分核路径）
    "mqa_c28": dict(n1=8, n2=1, head_dim=192, input_layout="TND", sparse_mode=0,
                    block_size=128, q_len=7, kv_len=4096, B=1, seed=28),
    # GQA (128,128,0) 常规 prefill，gS=2
    "C21": dict(n1=8, n2=4, head_dim=128, input_layout="TND", sparse_mode=0,
                block_size=128, q_len=64, kv_len=512, B=1, seed=21),
    # GQA (192,192,0) gS=8 中长 kv（call_pr_sink 模板 head_dim_v==head_dim_qk，
    # v≠qk 的 (192,128,0) 族需专用脚本，此处不覆盖）
    "C22": dict(n1=8, n2=1, head_dim=192, input_layout="TND",
                sparse_mode=0, block_size=128, q_len=7, kv_len=1024, B=1, seed=22),
    # GQA (64,64,0)，gS=4
    "C23": dict(n1=8, n2=2, head_dim=64, input_layout="TND", sparse_mode=0,
                block_size=128, q_len=16, kv_len=320, B=1, seed=23),
    # B=2 不等长（累计 q-len 契约 × 变长 kv；rope 分离族 (128,64,64) 需专用脚本，不在此覆盖）
    "C24": dict(n1=8, n2=4, head_dim=128, input_layout="TND", sparse_mode=0,
                block_size=128, q_len=7, kv_len=320, kv_len2=1152, B=2, seed=24),
}


def resolve(name):
    """case 名 → 规范名（保持原门禁调用形态 resolve(name)）。"""
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; known: {sorted(CASES)}")
    return name


def parse_case(name):
    c = dict(CASES[name])
    c.setdefault("v_head_dim", c["head_dim"])
    c.setdefault("rope_head_dim", 0)
    return c


def build_inputs(c):
    """构造 PA 池形态输入（与 call_pr_sink / call_baseline 的消费字段对齐）。

    B=1 用 c["kv_len"]；B≥2 用 c["kv_len"] / c["kv_len2"] 构造不等长 batch
    （TND 累计 q-len 契约的变长覆盖）。"""
    torch.manual_seed(c["seed"])
    B, T, D, vD = c["B"], c["q_len"], c["head_dim"], c["v_head_dim"]
    N1, N2, BS = c["n1"], c["n2"], c["block_size"]
    kv_lens = [c["kv_len"], c.get("kv_len2", c["kv_len"])] if B == 2 else [c["kv_len"]] * B
    blocks_per_req = [(kv + BS - 1) // BS for kv in kv_lens]
    pool = sum(blocks_per_req) + 2  # 预留 2 页余量

    q = (torch.rand(T * B, N1, D) * 2).bfloat16().npu()
    kpool = (torch.rand(pool, N2, BS, D) * 2).bfloat16().npu()
    vpool = (torch.rand(pool, N2, BS, vD) * 2).bfloat16().npu()
    # 每请求顺序编页（与官方逐请求 bit 对照时页序一致）
    pages = torch.arange(pool, dtype=torch.int32)
    bt = torch.zeros(B, max(blocks_per_req), dtype=torch.int32)
    cur = 0
    for i, nb in enumerate(blocks_per_req):
        bt[i, :nb] = pages[cur:cur + nb]
        cur += nb
    bt = bt.npu()

    qlens = [T] * B
    cum = torch.tensor([sum(qlens[:i + 1]) for i in range(B)], dtype=torch.int64).npu()
    kv_dev = torch.tensor(kv_lens, dtype=torch.int64).npu()

    return dict(q=q, kpool=kpool, vpool=vpool, bt=bt,
                qlen_dev=cum, kvlen_dev=kv_dev,
                qlen_host=list(cum.cpu().tolist()), kvlen_host=list(kv_lens),
                scale=1.0 / math.sqrt(D))


def bits_equal(out, ref):
    """bit 级对照 + mere（相对误差最大值）+ max_abs。"""
    assert out.shape == ref.shape and out.dtype == ref.dtype, \
        f"shape/dtype mismatch: {out.shape}/{out.dtype} vs {ref.shape}/{ref.dtype}"
    same_bits = torch.equal(out.view(torch.int16), ref.view(torch.int16)) \
        if out.dtype in (torch.bfloat16, torch.float16) else torch.equal(out, ref)
    o, r = out.float(), ref.float()
    diff = (o - r).abs()
    denom = r.abs().clamp_min(1e-6)
    mere = (diff / denom).max().item()
    max_abs = diff.max().item()
    return bool(same_bits), mere, max_abs
