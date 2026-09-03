"""FIA v2 sink GQA（kv_heads>1）测试用例构造器（plans/08 §3.2 W1，C21–C26 + 压测族）。

用例重构来源（不从零造）:
  - 形态骨架复用上游 case_builder.py 的 MLA 用例约定（TND 族布局 / PA 池 / 变长 kvlen /
    seed 确定性 / bf16 rand*2 分布），去掉 MLA 特化项（rope、512+64 拼接形态）;
  - 头数分组锚定上游真实模型形态（vllm-ascend attention_v1.py 消费形态:
    TND + page-attention KV + num_key_value_heads>1）:
      gS=2 ← Qwen3-0.6B (16q/8kv),  gS=4 ← Llama-3.1-8B (32q/8kv),
      gS=8 ← Qwen3-32B/GLM 类大模型 (96q/12kv 按 96 头压测口径);
  - 压测 shape 按 plans/08 §3.2 口径设计（96 q头 gSize=8 × kvlen 256K/512K 量级,
    对位 round_2 C19/C20 的 MLA 长序列门禁形态）。

与 MLA 用例（C01–C20 全部 n_kv_heads=1、D512+rope64）的关键差异:
  - kv_heads>1 → key/value 为独立张量（MLA 里 absorbed 形态二者共用同一池）;
  - rope 无输入 → query_rope/key_rope=None, rope_head_dim=0（AICPU CheckIsMla()=false → GQA 分支）;
  - head_dim ∈ {64,128,192}（非 512+64 拼接形态）;
  - 混合变长: 每请求 q token 数与 kvlen 均变长（MLA 用例固定 G tokens/req）。

双接口适配（host->device 迁移对比口径, 与 case_builder 完全一致）:
  - baseline: actual_seq_qlen/kvlen 传 Python list（官方 npu_fused_infer_attention_score_v2）
  - newop   : actual_seq_qlen/kvlen 传 device INT64 tensor（sink 两段式 entry）
用法:
  python3 gqa_case_builder.py --list
  python3 gqa_case_builder.py --check C21           # golden + 双算子 bit 对照（单进程内冒烟）
  python3 gqa_case_builder.py --shape C25           # 只打印形状
"""
import argparse, math, sys
import torch

# GQA 判定线与 MLA 门禁一致（BF16 2^-7; 见 case_builder.GOLDEN_MERE_TH）
GOLDEN_MERE_TH = 2.0 ** -7


def _cases():
    """有序用例表（等价 CSV 行; 字段语义见模块 docstring）。gate=1 为性能门禁/压测族。

    二进制包络约束（W1 首跑实测, 见 r3_w1_gqa_evidence.json 边界记录）:
      当前交付包 GQA tiling key 覆盖 = TND/NTD_TND(q 布局两种形态) + PA(BnNBsD)
      + rope_head_dim=0 + head_dim∈{64,128} + fp16/bf16 + sm=0;
      TND_NTD 组合布局与 D192(rope=0) 无对应 tiling key → 从门禁族移出、单列为边界项。
    """
    c = {}
    def add(name, scenario, n1, n2, d, qlen, kvlen, bs, layout, dtype="bfloat16", sparse_mode=0, gate=0, seed=20260910):
        assert n1 % n2 == 0, f"{name}: n1={n1} 不是 n2={n2} 整数倍"
        gs = n1 // n2
        assert gs > 1 or name.startswith(("mqa",)), f"{name}: 非法 GQA 用例 (gSize={gs})"
        c[name] = dict(case_name=name, scenario=scenario, n1=n1, n2=n2, gsize=gs, head_dim=d,
                       q_len_per_req=[int(x) for x in qlen.split("|")],
                       kv_len_per_req=[int(x) for x in kvlen.split("|")],
                       block_size=bs, input_layout=layout, dtype=dtype,
                       sparse_mode=sparse_mode, gate=gate, seed=seed)
    # ---- C21–C26 门禁族（gSize{2,4,6,8} × D{64,128} × TND/NTD_TND × sm=0 × 混合变长 × rope=None）----
    add("C21_gqa_gs2_d128_tnd",    "SigA_GQA_qwen06b", 16, 8, 128, "8|7|9|6|8|10|5|7",
        "77|129|130|4096|4097|16|128|8192", 128, "TND", seed=20260911)
    add("C22_gqa_gs4_d128_tnd",    "SigA_GQA_llama8b", 32, 8, 128, "8|8|7|9",
        "1024|1536|4096|12288", 128, "TND", dtype="float16", seed=20260912)
    add("C23_gqa_gs8_d64_tnd",     "SigA_GQA_small_d", 24, 3, 64, "6|8|5|9",
        "333|1000|4096|8192", 128, "TND", seed=20260913)
    add("C24_gqa_gs2_d128_ntdtnd", "SigA_GQA_ntd_arm", 16, 8, 128, "8|9",
        "777|4999", 128, "NTD_TND", seed=20260914)
    add("C25_gqa_gs6_mixvarlen",   "SigA_GQA_batch_mix", 48, 8, 128, "9|6|11|4|8|7",
        "61|2000|128|4097|999|5120", 128, "TND", seed=20260915)
    add("C26_gqa_gs8_maxmix",      "SigA_GQA_stress_seed", 96, 12, 128, "8|7|9|6",
        "1500|7000|2048|16384", 128, "TND", dtype="float16", seed=20260916)
    # ---- V4 性能门禁压测族（kernel-bound：96 q头 gS=8 × 256K/512K，对位 C19/C20 口径）----
    add("S1_gqa_r96_gs8_kv256k", "LongKV_GQA_r96", 96, 12, 128, "8",
        "262144", 128, "TND", gate=1, seed=20260917)
    add("S2_gqa_r96_gs8_kv512k", "LongKV_GQA_r96", 96, 12, 128, "8",
        "524288", 128, "TND", gate=1, seed=20260918)
    # ---- C27/C28（2026-08-28 整网接入边界复现: GLM-5.2 DSpark draft per-rank 形态）----
    # 服务实况（vllm TP8 切分后每卡）: N1=N2=8 → MHA gSize=1, head_dim=192, TND, bf16,
    # sm=0, rope=0, PA 池; B=1/qlen=7/kvlen=20 按首请求现场参数。
    # 命名说明: g==1 触发本构造器既有 assert（非 mqa 前缀拒绝）, 按既有约定以 mqa_ 开头
    # 携带 C27/C28 序号（铁律: 只新增用例, 不改既有语义）。
    # C27 = 复现整网 aclnn 失败形态; C28 = D128 单轴隔离（判 g=1 本身是否有 tiling key）。
    add("mqa_c27_g1_d192_tnd", "SigA_DSparkDraft_g1_d192", 8, 8, 192, "7", "20", 128, "TND", seed=20260919)
    add("mqa_c28_g1_d128_tnd", "SigA_DSparkDraft_g1_d128", 8, 8, 128, "7", "20", 128, "TND", seed=20260920)
    return c


def resolve(name):
    cs = _cases()
    if name in cs:
        return cs[name]
    hit = [k for k in cs if k.startswith(name)]
    assert len(hit) == 1, f"case '{name}' 匹配 {hit}"
    return cs[hit[0]]


def _dtype(s):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def _scale_of(d):
    return 1.0 / math.sqrt(d)


def parse_case(spec):
    c = spec["case_name"]
    b = len(spec["q_len_per_req"])
    assert b == len(spec["kv_len_per_req"]), f"{c}: q/kv 请求数不一致"
    case = dict(spec)
    case["B"] = b
    case["T"] = sum(case["q_len_per_req"])
    bt_w = max(math.ceil(x / spec["block_size"]) for x in case["kv_len_per_req"])
    pool = sum(math.ceil(x / spec["block_size"]) for x in case["kv_len_per_req"]) + 2  # 余 2 块校验寻址正确性
    case["bt_w"], case["pool"] = bt_w, pool
    return case


def build_inputs(c, device="npu:0", require_npu=True):
    """确定性构造一例 GQA 输入。key/value 独立张量（GQA: kv_heads>1，非 MLA absorbed 共用）。"""
    if require_npu:
        import torch_npu  # noqa: F401
    g = torch.Generator().manual_seed(c["seed"])
    B, N1, N2, D, BS = c["B"], c["n1"], c["n2"], c["head_dim"], c["block_size"]
    T, dt = c["T"], _dtype(c["dtype"])

    q_lin = (torch.rand(T, N1, D, generator=g) * 2).to(dt)
    # NTD_TND 的 query 存储为头主序 [N1,T,D]; 其余（TND 族）token 主序 [T,N1,D]
    q = q_lin.permute(1, 0, 2).contiguous() if c["input_layout"] == "NTD_TND" else q_lin
    kpool = (torch.rand(c["pool"], N2, BS, D, generator=g) * 2).to(dt)
    vpool = (torch.rand(c["pool"], N2, BS, D, generator=g) * 2).to(dt)

    # block_table: 每请求顺序连续且互不重叠的 block id, 余列补 0（与 case_builder 同构）
    bt = torch.zeros(B, c["bt_w"], dtype=torch.int32)
    start = 0
    for i, lv in enumerate(c["kv_len_per_req"]):
        nb = math.ceil(lv / BS)
        bt[i, :nb] = torch.arange(start, start + nb, dtype=torch.int32)
        start += nb
    assert start <= c["pool"]

    qlen_cum = []
    acc = 0
    for x in c["q_len_per_req"]:
        acc += x
        qlen_cum.append(acc)

    def to(t):
        return t.to(device) if require_npu else t

    return dict(
        case=c, q=to(q), kpool=to(kpool), vpool=to(vpool), bt=to(bt),
        qlen_host=qlen_cum, kvlen_host=list(c["kv_len_per_req"]),
        qlen_dev=to(torch.tensor(qlen_cum, dtype=torch.int64)),
        kvlen_dev=to(torch.tensor(c["kv_len_per_req"], dtype=torch.int64)),
        scale=_scale_of(D),
    )


def call_baseline(inp, return_lse=False):
    """基线: 官方 npu_fused_infer_attention_score_v2 GQA 路径（seq len 走 host list, 保持原样口径）。"""
    import torch_npu
    c = inp["case"]
    kw = dict(
        num_query_heads=c["n1"], num_key_value_heads=c["n2"],
        input_layout=c["input_layout"], sparse_mode=c["sparse_mode"],
        softmax_scale=inp["scale"], block_table=inp["bt"], block_size=c["block_size"],
        actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
        return_softmax_lse=return_lse,
    )
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["vpool"], **kw)


def call_newop(inp, return_lse=True):
    """新算子: sink 两段式 entry（device tensor 传参），query_rope/key_rope=None 走 GQA 分支。"""
    sys.path.insert(0, __import__("os").path.join(
        __import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "python"))
    from fia_v2_sink_entry import entry
    c = inp["case"]
    kw = dict(
        query_rope=None, key_rope=None,
        num_query_heads=c["n1"], num_key_value_heads=c["n2"],
        input_layout=c["input_layout"], input_layout_kv="PA",
        sparse_mode=c["sparse_mode"], softmax_scale=inp["scale"],
        block_table=inp["bt"], block_size=c["block_size"],
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        return_softmax_lse=return_lse,
    )
    return entry(inp["q"], inp["kpool"], inp["vpool"], **kw)


@torch.no_grad()
def golden(inp):
    """fp32 参考（CPU）: GQA 连续头分组（h = n2*gSize + j 组内序）、PA 分页 gather、无掩码全量。

    返回 output 布局的 [T,N1,D] 形态:
      TND 输入 → out [T,N1,D]; NTD_TND 输入(q=[N1,T,D]) → out 同为 [T,N1,D]。
      （TND_NTD 的输出为 [N1,T,D]，当前二进制不支持，见包络边界记录。）
    """
    c = inp["case"]
    B, N1, N2, D, GS, BS = c["B"], c["n1"], c["n2"], c["head_dim"], c["gsize"], c["block_size"]
    q = inp["q"].cpu().float() if inp["q"].is_npu else inp["q"].float()
    if c["input_layout"] == "NTD_TND":
        q = q.permute(1, 0, 2).contiguous()          # [N1,T,D] -> [T,N1,D]
    kp = inp["kpool"].cpu() if inp["kpool"].is_npu else inp["kpool"]
    vp = inp["vpool"].cpu() if inp["vpool"].is_npu else inp["vpool"]
    btc = inp["bt"].cpu() if inp["bt"].is_npu else inp["bt"]

    o = torch.zeros(c["T"], N1, D)
    t_off = 0
    for i, lv in enumerate(c["kv_len_per_req"]):
        nb = math.ceil(lv / BS)
        ids = btc[i, :nb].long()
        kl = kp.index_select(0, ids).reshape(-1, N2, D)[:lv].permute(1, 0, 2).float()   # [N2,L,D]
        vl = vp.index_select(0, ids).reshape(-1, N2, D)[:lv].permute(1, 0, 2).float()
        gtok = c["q_len_per_req"][i]
        qs = q[t_off:t_off + gtok].view(gtok, N2, GS, D)                                # 连续头分组
        s = torch.einsum("tnjd,nld->njtl", qs, kl) * inp["scale"]                       # [N2,gtok,L]
        p = torch.softmax(s, dim=-1)
        og = torch.einsum("njtl,nld->tnjd", p, vl).reshape(gtok, N1, D)
        o[t_off:t_off + gtok] = og
        t_off += gtok
    return o


def mere(a, b):
    d = (a.float() - b.float()).abs()
    return (d / b.float().abs().clamp_min(1e-6)).mean().item(), d.max().item()


def bits_equal(a, b):
    """R6 bit 级判定（cos 不敏感判定禁用）。"""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, None, None
    m, mx = mere(a.cpu(), b.cpu())
    eq = bool(torch.equal(a.cpu(), b.cpu()))
    return eq, m, mx


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", nargs="*", default=None)
    ap.add_argument("--shape", nargs="*", default=None)
    a = ap.parse_args()
    cs = _cases()
    if a.list or not sys.argv[1:]:
        for n, raw in cs.items():
            print(f"{n:26s} n1={raw['n1']:2d} n2={raw['n2']:2d} gS={raw['n1']//raw['n2']:2d} "
                  f"D={raw['head_dim']:3d} {raw['input_layout']:8s} {raw['dtype']} "
                  f"{'<GATE>' if raw['gate'] else ''}")
    for n in a.shape or []:
        c = parse_case(resolve(n))
        inp = build_inputs(c, require_npu=False)
        print(f"[{n}] q{tuple(inp['q'].shape)} pool k/v{tuple(inp['kpool'].shape)} "
              f"bt{tuple(inp['bt'].shape)} qlen={inp['qlen_host']} kvlen={inp['kvlen_host']}")
    if a.check is not None:
        ok_all = True
        for n in a.check:
            c = parse_case(resolve(n))
            inp = build_inputs(c)
            out_n, lse_n = call_newop(inp, return_lse=True)
            out_b, lse_b = call_baseline(inp, return_lse=True)
            eq, m, mx = bits_equal(out_n, out_b)
            ref = golden(inp).to(out_n.device)
            mg, mgx = mere(out_n, ref)
            ok = eq and mg < GOLDEN_MERE_TH
            ok_all &= ok
            print(f"[{n}] shape={tuple(out_n.shape)} bit_exact={eq} mere_vs_base={m} max_abs={mx} "
                  f"MERE_golden={mg:.6f} PASS={ok}")
        sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    _main()
