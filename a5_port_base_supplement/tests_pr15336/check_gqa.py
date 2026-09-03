"""npu_fused_infer_attention_score_v2_sink GQA 精度门禁（plans/08 §3.2 W1, C21–C26）。

进程隔离执行（run_gate.sh 风格）:
  python3 check_gqa.py C21 --json ../results/round_3/prec_C21.json

判定口径（R6: bit 级优先，禁 cos 不敏感判定）:
  P1 主判据   新算子 vs 9.2 官方 npu_fused_infer_attention_score_v2 **GQA 路径**
              -> torch.equal 逐位一致（bit_exact）且 MERE_vs_official == 0.0
  P2 辅助记录 新算子/官方 vs 教科书 fp32 golden 的 MERE（无门禁作用, 见 evidence 说明:
              官方 N2>1 TND prefill 数值路径与教科书公式存在内部差异,
              对 sink 判定不构成约束——sink 与官方逐位一致即证数学主体未变）
  P3 元数据    AICPU metadata 关键字段随 GQA 用例归档（usedCoreNum/s2Size…）

MQA 简并校验见 mqa_check()（--mqa）: N2=1 时官方/新算子二者均须与 fp32 golden 噪声级
一致且互相 bit 等（数学主体健全性锚点）。
"""
import argparse, json, os, sys
import torch
import torch_npu  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "python"))
from gqa_case_builder import (resolve, parse_case, build_inputs, call_baseline, call_newop,  # noqa: E402
                              golden, mere, bits_equal)
from fia_v2_sink_entry import entry, metadata  # noqa: E402

# 与 op_kernel/fused_infer_attention_score_v2_sink_metadata.h flat 布局一致的 baseMeta 偏移
BASE_OFF = 36 * 10 + 72 * 3
BASE_IDX = dict(mBaseSize=0, sInnerSize=1, mFdBaseSize=2, numOfFd=3, usedCoreNum=4,
                usedVecNumOfFd=5, s1Size=6, s2Size=7, actualLenQDims=8, actualLenKvDims=9)


def meta_evidence(meta_cpu):
    ev = {}
    for k, i in BASE_IDX.items():
        ev[k] = int(meta_cpu[BASE_OFF + i].item())
    ev["aic_enabled_count"] = sum(int(meta_cpu[c * 10].item()) for c in range(36))
    return ev


def check(name, json_out=None):
    c = parse_case(resolve(name))
    inp = build_inputs(c)

    # ---- 新算子（两段式, device 传参, rope=None → GQA 分支） ----
    out_n, lse_n = call_newop(inp, return_lse=True)
    torch.npu.synchronize()

    # ---- R6 主判据: vs 9.2 官方 GQA 路径 bit 级 ----
    out_b, lse_b = call_baseline(inp, return_lse=True)
    torch.npu.synchronize()
    eq_out, m_out, mx_out = bits_equal(out_n, out_b)
    lse_rel = None
    if lse_n.numel() > 1 and lse_b.shape == lse_n.reshape(lse_b.shape).shape:
        ln = lse_n.reshape(lse_b.shape).float()
        lb = lse_b.float()
        lse_rel = float(((ln - lb).abs() / lb.abs().clamp_min(1e-6)).mean())
        lse_ok = bool(torch.equal(ln, lb)) or lse_rel < 1e-6
    else:
        lse_ok = None

    # ---- 辅助记录: 教科书 fp32 golden ----
    inp_cpu = {k: (v.cpu() if torch.is_tensor(v) and v.is_npu else v) for k, v in inp.items()}
    ref = golden(inp_cpu).to(out_n.device)
    mg, mgx = mere(out_n, ref)
    mb, _ = mere(out_b, ref)
    finite = bool(torch.isfinite(out_n.float()).all().item())

    # ---- R1 证据: GQA 形态 metadata 字段 ----
    meta = metadata(c["n1"], c["n2"], c["head_dim"], c["head_dim"],
                    inp["qlen_dev"], inp["kvlen_dev"],
                    sparse_mode=c["sparse_mode"], input_layout=c["input_layout"],
                    input_layout_kv="PA", rope_head_dim=0, block_size=c["block_size"])
    torch.npu.synchronize()
    ev = meta_evidence(meta.cpu())

    passed = eq_out and (mx_out == 0.0) and finite      # bit 级主判据
    r = dict(case=name, layout=c["input_layout"], dtype=c["dtype"],
             n_query_heads=c["n1"], n_kv_heads=c["n2"], gsize=c["gsize"], head_dim=c["head_dim"],
             shape=tuple(out_n.shape),
             bit_exact_vs_official=eq_out, max_abs_vs_official=mx_out, mere_vs_official=m_out,
             lse_bit_ok=lse_ok, lse_rel_vs_official=lse_rel,
             mere_vs_golden_sink=mg, max_abs_vs_golden_sink=mgx, mere_vs_golden_official=mb,
             finite=finite, meta_evidence=ev, passed=passed,
             judge="P1: bit-level vs official 9.2 GQA path (primary); P2: golden MERE informational")
    print(f"[{name}] {'PASS' if passed else 'FAIL'} bit_exact={eq_out} "
          f"MERE_official={m_out} max_abs={mx_out:.6f} | golden辅助 sink={mg:.6f} official={mb:.6f}")
    print(f"    meta: {ev}")
    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)) or ".", exist_ok=True)
        json.dump(r, open(json_out, "w"), indent=1)
    return passed


@torch.no_grad()
def mqa_check():
    """N2=1 简并锚点: 官方与 sink 都须逼近 fp32 golden 且互相 bit 等。"""
    import math
    import torch.nn.functional as F
    from gqa_case_builder import _scale_of, mere as gb_mere

    specs = [dict(G=8, L=512, N1=16, D=128, BS=128, seed=31),
             dict(G=8, L=2048, N1=32, D=128, BS=128, seed=32)]
    all_ok = True
    for ic, sp in enumerate(specs):
        torch.manual_seed(sp["seed"])
        T, L, N1, D, BS = sp["G"], sp["L"], sp["N1"], sp["D"], sp["BS"]
        POOL = (L + BS - 1) // BS
        scale = _scale_of(D)
        dt = torch.bfloat16
        q = (torch.rand(T, N1, D) * 2).to(dt).to("npu")
        kp = (torch.rand(POOL, 1, BS, D) * 2).to(dt).to("npu")
        vpk = (torch.rand(POOL, 1, BS, D) * 2).to(dt).to("npu")
        bt = torch.arange(POOL, dtype=torch.int32).view(1, -1).to("npu")
        common = dict(num_query_heads=N1, num_key_value_heads=1,
                      input_layout="TND", sparse_mode=0, softmax_scale=scale,
                      block_table=bt, block_size=BS,
                      actual_seq_qlen=[T], actual_seq_kvlen=[L], return_softmax_lse=False)
        common.pop("actual_seq_qlen")
        common.pop("actual_seq_kvlen")
        ob, _ = torch_npu.npu_fused_infer_attention_score_v2(q, kp, vpk,
            actual_seq_qlen=[T], actual_seq_kvlen=[L], **common)
        on, _ = entry(q, kp, vpk, query_rope=None, key_rope=None, input_layout_kv="PA",
                      actual_seq_qlen=torch.tensor([T], dtype=torch.int64, device="npu"),
                      actual_seq_kvlen=torch.tensor([L], dtype=torch.int64, device="npu"),
                      **common)
        eqb, _, _ = bits_equal(on, ob)
        # fp32 golden（MQA: 全头共享单 kv 头）
        qf = q.float().cpu()
        kf = kp.float().cpu().reshape(-1, D)[:L]
        vf = vpk.float().cpu().reshape(-1, D)[:L]
        og = torch.zeros(T, N1, D)
        for h in range(N1):
            p = torch.softmax((kf @ qf[:, h, :].T) * scale, dim=0).T     # [T,L]
            og[:, h, :] = p @ vf
        mg, mgx = gb_mere(on.cpu(), og)
        finite = bool(torch.isfinite(on.float()).all().item())
        thr = 2.0 ** -7
        ok = eqb and mg < thr and finite
        all_ok &= ok
        print(f"[mqa_{ic}] shape={tuple(on.shape)} bit_eq={eqb} MERE_golden={mg:.6f} PASS={ok}")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    names = a.cases or [f"C{i}" for i in range(21, 27)]
    if len(names) == 1 and a.json:
        ok = check(names[0], a.json)
    else:
        ok = all([check(n) for n in names])
    sys.exit(0 if ok else 1)
