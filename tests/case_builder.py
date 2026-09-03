"""C-series case builder（MLA）— a5_port migration rebuild (2026-09-02).

原版未随交接包携带；按 pr15336_gate_c01.py 的消费契约重建：
  load_cases() → resolve(cases, "C01") → parse_case(c) → build_inputs(c)
C01 = MLA absorbed 形态（qk=v=512 + rope=64，n2=1），TND 累计 q-len + PA 池。
参数为迁移重建值（接口文档 §6 MLA 行的代表性形态）。
"""
import torch

_CASES = {
    # MLA absorbed：D=512(nope), R=64(rope), kvh=1；H 取 8（代表性 gS=8）
    "C01": dict(H=8, kvh=1, D=512, R=64, BS=128, sparse_mode=0,
                q_len=7, kv_len=640, B=1, seed=101),
}


def load_cases():
    return dict(_CASES)


def resolve(cases, name):
    if name not in cases:
        raise KeyError(f"unknown case {name!r}; known: {sorted(cases)}")
    return name


def parse_case(name):
    return dict(_CASES[name])


def build_inputs(c):
    torch.manual_seed(c["seed"])
    B, T, BS = c["B"], c["q_len"], c["BS"]
    H, kvh, D, R = c["H"], c["kvh"], c["D"], c["R"]
    nblocks = (c["kv_len"] + BS - 1) // BS
    pool = nblocks * B + 2

    q = (torch.rand(T * B, H, D) * 2).bfloat16().npu()
    qpe = (torch.rand(T * B, H, R) * 2).bfloat16().npu()
    kpool = (torch.rand(pool, kvh, BS, D) * 2).bfloat16().npu()
    krpool = (torch.rand(pool, kvh, BS, R) * 2).bfloat16().npu()
    bt = torch.arange(pool, dtype=torch.int32).view(B, -1).npu()

    qlens = [T] * B
    cum = torch.tensor([sum(qlens[:i + 1]) for i in range(B)], dtype=torch.int64).npu()
    kv_dev = torch.tensor([c["kv_len"]] * B, dtype=torch.int64).npu()

    return dict(q=q, qpe=qpe, kpool=kpool, krpool=krpool, bt=bt,
                qlen_dev=cum, kvlen_dev=kv_dev, mask=None,
                qlen_host=list(cum.cpu().tolist()), kvlen_host=[c["kv_len"]] * B)
