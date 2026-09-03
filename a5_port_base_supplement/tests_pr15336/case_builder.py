"""FIA v2 (npu_fused_infer_attention_score_v2) 测试用例构造器。

用例规格: fia_v2_cases.csv (11 条: C01-C10 常规 + C11 门禁/压测)
约定 (已在 Ascend910 本机冒烟验证, MERE=0.0015 @ C03 mini):
  - query TND [T,H,512] / query_rope [T,H,64], KV 池 NTD [pool,1,128,512/64], value 与 key 同一 tensor
  - actual_seq_qlen = 每 request token 数累积和 (cumsum); actual_seq_kvlen = 每 request 真实上下文长
  - 掩码语义: suffix-causal —— 第 i 请求第 j 个 query token 关注 kv r ∈ [0, kvlen_i - G + j]
  - softmax_scale = 1/sqrt(192); 数值分布 rand*2 (正分布)
双接口适配 (host->device 迁移对比口径):
  - baseline: actual_seq_qlen/kvlen 传 Python list (host, 保持原样)
  - newop   : actual_seq_qlen/kvlen 传 device INT64 tensor (新算子)
用法:
  python3 case_builder.py --list
  python3 case_builder.py --check C03 C04        # 基线 vs fp32 golden
  python3 case_builder.py --shape C11            # 只打印构造形状(不上板)
"""
import argparse, csv, math, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "fia_v2_cases.csv")
SCALE = 1.0 / math.sqrt(192.0)          # qk_head_dim = 128 + 64
LAYOUT = "TND_NTD"
GOLDEN_MERE_TH = 2.0 ** -7              # BF16 判定线


def load_cases():
    cases = {}
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            cases[row["case_name"]] = row
    return cases


def resolve(names_keys, name):
    """支持全名或唯一前缀 (如 C03 -> C03_sig_a_min)。"""
    if name in names_keys:
        return name
    hit = [k for k in names_keys if k.startswith(name)]
    assert len(hit) == 1, f"case '{name}' 匹配 {hit}"
    return hit[0]


def parse_case(row):
    kv = [int(x) for x in row["kv_len_per_req"].split("|")]
    c = dict(
        name=row["case_name"], scenario=row["scenario"],
        B=int(row["batch_size"]), G=int(row["tokens_per_req"]),
        H=int(row["n_head_slots"]), kvh=int(row["n_kv_heads"]),
        D=int(row["kv_lora_rank"]), R=int(row["rope_dim"]), BS=int(row["block_size"]),
        kvlen=kv, pool=int(row["pool_blocks"]), bt_w=int(row["bt_width"]),
        use_mask=row["use_mask"] == "1", sparse_mode=int(row["sparse_mode"]),
        gate=row["gate"] == "1", seed=int(row["seed"]),
    )
    c["T"] = c["B"] * c["G"]
    if c["bt_w"] == 0:
        c["bt_w"] = max(1, math.ceil(max(c["kvlen"]) / c["BS"]))
    need = sum(math.ceil(x / c["BS"]) for x in c["kvlen"])
    assert need <= c["pool"], f"{c['name']}: blocks need {need} > pool {c['pool']}"
    if c["use_mask"]:
        c["mask_rows"], c["mask_cols"] = int(row["mask_rows"]), int(row["mask_cols"])
    return c


def build_inputs(c, device="npu:0", require_npu=True):
    """确定性构造一例输入 (seed 由 CSV 给定)。返回 dict, 含 baseline/newop 双接口所需全部张量。"""
    if require_npu:
        import torch_npu  # noqa: F401
    g = torch.Generator().manual_seed(c["seed"])
    B, G, H, D, R, BS = c["B"], c["G"], c["H"], c["D"], c["R"], c["BS"]
    T = c["T"]

    q = (torch.rand(T, H, D, generator=g) * 2).to(torch.bfloat16)
    qpe = (torch.rand(T, H, R, generator=g) * 2).to(torch.bfloat16)
    kpool = (torch.rand(c["pool"], 1, BS, D, generator=g) * 2).to(torch.bfloat16)
    krpool = (torch.rand(c["pool"], 1, BS, R, generator=g) * 2).to(torch.bfloat16)

    # block_table: 每请求顺序取连续且互不重叠的 block id, 余下列补 0
    bt = torch.zeros(B, c["bt_w"], dtype=torch.int32)
    start = 0
    for i, L in enumerate(c["kvlen"]):
        nb = math.ceil(L / BS)
        bt[i, :nb] = torch.arange(start, start + nb, dtype=torch.int32)
        start += nb
    assert start <= c["pool"]

    qlen_cum = [G * (i + 1) for i in range(B)]          # 累积和 (与 vllm mla_v1.py:851 一致)

    def to(t):
        return t.to(device) if require_npu else t

    out = dict(
        case=c, q=to(q), qpe=to(qpe), kpool=to(kpool), krpool=to(krpool), bt=to(bt),
        qlen_host=qlen_cum, kvlen_host=list(c["kvlen"]),
        qlen_dev=to(torch.tensor(qlen_cum, dtype=torch.int64)),
        kvlen_dev=to(torch.tensor(c["kvlen"], dtype=torch.int64)),
        scale=SCALE,
    )
    if c["use_mask"]:
        out["mask"] = to(torch.triu(torch.ones(c["mask_rows"], c["mask_cols"], dtype=torch.int8), diagonal=1))
    return out


def call_baseline(inp, return_lse=False):
    """基线算子: torch_npu 官方 npu_fused_infer_attention_score_v2, seq len 走 host list (保持原样)。"""
    import torch_npu
    c = inp["case"]
    kw = dict(
        query_rope=inp["qpe"], key_rope=inp["krpool"],
        num_query_heads=c["H"], num_key_value_heads=c["kvh"],
        input_layout=LAYOUT, sparse_mode=c["sparse_mode"],
        softmax_scale=inp["scale"], block_table=inp["bt"], block_size=c["BS"],
        actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
        return_softmax_lse=return_lse,
    )
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)


@torch.no_grad()
def golden(inp):
    """fp32 参考实现 (CPU, 向量化): MLA absorbed MQA + rope + suffix-causal + paged gather。
    返回 [H, T, D] (NTD)。要求所有 kvlen_i >= G (CSV 已保证)。"""
    c = inp["case"]
    B, G, H, D, R, BS = c["B"], c["G"], c["H"], c["D"], c["R"], c["BS"]
    q, qpe = inp["q"].float(), inp["qpe"].float()
    kpool = inp["kpool"].float().reshape(-1, BS, D)
    krpool = inp["krpool"].float().reshape(-1, BS, R)
    btc = inp["bt"] if not inp["bt"].is_npu else inp["bt"].cpu()
    o = torch.zeros(H, c["T"], D)
    for i, L in enumerate(c["kvlen"]):
        nb = math.ceil(L / BS)
        ids = btc[i, :nb].long()                        # 跳过 0 填充列
        klat = kpool[ids].reshape(-1, D)[:L]            # [L, D] (块对齐后截断)
        krope = krpool[ids].reshape(-1, R)[:L]
        for j in range(G):
            rows = slice(i * G + j, i * G + j + 1)
            ql, qp = q[rows].squeeze(0), qpe[rows].squeeze(0)     # [H,D]/[H,R]
            s = (ql @ klat.T + qp @ krope.T) * inp["scale"]       # [H, L]
            if c["sparse_mode"] != 0:       # sm=3: suffix-causal; sm=0: 无掩码全量 (实测 MERE 0.0015)
                lim = L - G + j + 1
                s[:, lim:] = float("-inf")
            p = torch.softmax(s, dim=-1)
            o[:, rows] = (p @ klat).unsqueeze(1)                   # [H, 1, D]
    return o


def mere(a, b):
    d = (a.float() - b.float()).abs()
    return (d / b.float().abs().clamp_min(1e-6)).mean().item(), d.max().item()


def check(name):
    cases = load_cases()
    c = parse_case(cases[resolve(cases, name)])
    inp = build_inputs(c)
    out, _lse = call_baseline(inp)
    ref = golden(inp).to(out.device)
    m, mx = mere(out, ref)
    ok = m < GOLDEN_MERE_TH and torch.isfinite(out.float()).all().item()
    print(f"[{name}] shape={tuple(out.shape)} MERE={m:.6f} max_abs={mx:.6f} "
          f"golden={'PASS' if ok else 'FAIL'} (th={GOLDEN_MERE_TH:.5f})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", nargs="*", default=None)
    ap.add_argument("--shape", nargs="*", default=None)
    a = ap.parse_args()
    cases = load_cases()
    if a.list or not sys.argv[1:]:
        for n, r in cases.items():
            c = parse_case(r)
            tag = " <GATE>" if c["gate"] else ""
            print(f"{n:20s} B={c['B']:2d} G={c['G']} H={c['H']:2d} kvlen={r['kv_len_per_req']:35s} "
                  f"pool={c['pool']:5d} bt=[{c['B']},{c['bt_w']}] mask={'Y' if c['use_mask'] else 'N'}{tag}")
    for n in a.shape or []:
        c = parse_case(cases[resolve(cases, n)])
        inp = build_inputs(c, require_npu=False)
        print(f"[{n}] q{tuple(inp['q'].shape)} qpe{tuple(inp['qpe'].shape)} "
              f"pool{tuple(inp['kpool'].shape)} bt{tuple(inp['bt'].shape)} "
              f"qlen{inp['qlen_host']} kvlen{inp['kvlen_host']}")
    if a.check is not None:
        names = [resolve(cases, n) for n in (a.check or list(cases))]
        all_ok = all([check(n) for n in names])
        sys.exit(0 if all_ok else 1)
