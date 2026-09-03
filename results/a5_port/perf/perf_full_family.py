"""FIA v2 Sink（A5）全族补测：MLA C04–C08/C10–C20 + GQA C21–C26，精度+性能双栏。

用法：
  cd /home/t00886357/a5_port_base_supplement/tests_pr15336
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=2 FIA_GATE_TREE=/home/t00886357/fia_sink_a5_port_base/vllm-ascend
  OPP=$FIA_GATE_TREE/vllm_ascend/_cann_ops_custom
  export ASCEND_CUSTOM_OPP_PATH=$OPP:$OPP/vendors/custom_transformer
  python3 perf_full_family.py [--json out.json]

口径：
- 精度：sink vs 官方同参对照（MLA 带 rope/mask，scale=0.0721687836487032 与 C01 口径连续；
  官方拒收形态标 N/A）
- 性能：sink 两段式完整成本 vs 官方单调用，warmup 2 + 计时（大 case 10 轮/小 case 20 轮）中位
- 大张量逐 case 释放（empty_cache）防 HBM 累积
"""
import argparse
import glob
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import time

import torch
import torch_npu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "a5_port_base_supplement", "tests_pr15336"))

from case_builder import load_cases as mla_cases, parse_case as mla_parse, build_inputs as mla_build  # noqa: E402
from gqa_case_builder import resolve as gqa_resolve, parse_case as gqa_parse, build_inputs as gqa_build  # noqa: E402

MLA_SCALE = 0.0721687836487032  # 与 pr15336_gate_c01 口径连续


def load_ext(pr_tree):
    c = sorted(glob.glob(os.path.join(pr_tree, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
    ldr = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", c)
    spec = importlib.util.spec_from_loader("vllm_ascend_C", ldr)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mla_sink(inp, c):
    N1, N2, D, R, BS = c["H"], c["kvh"], c["D"], c["R"], c["BS"]
    B = inp["kvlen_dev"].shape[0]
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)
    out, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=meta, num_query_heads=N1,
        num_key_value_heads=N2, softmax_scale=MLA_SCALE, input_layout="TND",
        sparse_mode=c["sparse_mode"], block_size=BS)
    return out


def mla_official(inp, c):
    N1, N2, BS = c["H"], c["kvh"], c["BS"]
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        num_query_heads=N1, num_key_value_heads=N2, input_layout="TND",
        sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
        block_table=inp["bt"], block_size=BS,
        actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
        return_softmax_lse=False)
    return out


def gqa_sink(inp, c):
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        c["n1"], c["n2"], c["head_dim"], c["head_dim"],
        actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(),
        batch_size=inp["kvlen_dev"].shape[0], sparse_mode=c["sparse_mode"],
        input_layout=c["input_layout"], input_layout_kv="PA",
        rope_head_dim=0, block_size=c["block_size"],
        aic_core_num=28, aiv_core_num=56)
    out, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["vpool"], actual_seq_qlen=inp["qlen_dev"],
        actual_seq_kvlen=inp["kvlen_dev"], block_table=inp["bt"], meta_data=meta,
        num_query_heads=c["n1"], num_key_value_heads=c["n2"],
        softmax_scale=inp["scale"], input_layout=c["input_layout"],
        sparse_mode=c["sparse_mode"], block_size=c["block_size"])
    return out


def gqa_official(inp, c):
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        inp["q"], inp["kpool"], inp["vpool"], num_query_heads=c["n1"],
        num_key_value_heads=c["n2"], input_layout=c["input_layout"],
        sparse_mode=c["sparse_mode"], softmax_scale=inp["scale"],
        block_table=inp["bt"], block_size=c["block_size"],
        actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
        return_softmax_lse=False)
    return out


def accuracy(sink, ref):
    if ref is None:
        return {"bit": None, "mere": None, "max_abs": None, "note": "official N/A"}
    same = torch.equal(sink.cpu(), ref.cpu())
    d = (sink.float().cpu() - ref.float().cpu()).abs()
    return {"bit": bool(same),
            "mere": (d / ref.float().cpu().abs().clamp_min(1e-6)).max().item(),
            "max_abs": d.max().item()}


def bench(fn, *a, warmup=2, iters=20):
    for _ in range(warmup):
        fn(*a)
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*a)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "perf_full_family.json"))
    ap.add_argument("--only", default=None,
                    help="单 case 模式：只跑该名字（MLA 全名或 GQA 短名），"
                         "输出一行 RESULT:{json} 供外层收集（进程隔离防挂死污染）")
    args = ap.parse_args()
    load_ext(os.environ.get("FIA_GATE_TREE",
                            "/home/t00886357/fia_sink_a5_port_base/vllm-ascend"))

    results = {}
    mla = mla_cases()
    mla_names = [n for n in mla if any(n.startswith(f"C{i:02d}_") for i in
                                       list(range(4, 9)) + list(range(10, 21)))]
    if args.only:
        mla_names = [n for n in mla_names if n == args.only]
    print(f"{'case':<22}{'bit':>6}{'mere':>11}{'sink(ms)':>10}{'off(ms)':>9}{'ratio':>8}")

    for name in mla_names:
        c = mla_parse(mla[name])
        inp = mla_build(c)
        total_kv = sum(c["kvlen"])
        iters = 10 if total_kv > 100000 else 20
        r = None
        try:
            s = mla_sink(inp, c)
            torch.npu.synchronize()
            try:
                r = mla_official(inp, c)
                torch.npu.synchronize()
            except Exception:
                r = None
            acc = accuracy(s, r)
            s_ms = bench(mla_sink, inp, c, iters=iters)
            o_ms = bench(mla_official, inp, c, iters=iters) if r is not None else None
        except Exception as e:
            acc, s_ms, o_ms = {"bit": None, "mere": None, "max_abs": None,
                               "note": f"sink fail: {str(e)[:60]}"}, None, None
        ratio = s_ms / o_ms if (s_ms and o_ms) else None
        results[name] = {"acc": acc, "sink_ms": s_ms, "official_ms": o_ms, "ratio": ratio}
        m = acc["mere"]
        print(f"{name:<22}{str(acc['bit']):>6}{(f'{m:.2e}' if m is not None else 'N/A'):>11}"
              f"{(f'{s_ms:.3f}' if s_ms else 'N/A'):>10}"
              f"{(f'{o_ms:.3f}' if o_ms else 'N/A'):>9}"
              f"{(f'{ratio:.3f}' if ratio else 'N/A'):>8}")
        try:
            torch.npu.empty_cache()
        except Exception:
            pass

    gqa_names = ["C21", "C22", "C23", "C24", "C25", "C26"]
    if args.only:
        gqa_names = [n for n in gqa_names if "GQA_" + n == args.only]
    for name in gqa_names:
        c = gqa_parse(gqa_resolve(name))
        inp = gqa_build(c)
        total_kv = sum(c["kv_len_per_req"])
        iters = 10 if total_kv > 100000 else 20
        try:
            s = gqa_sink(inp, c)
            torch.npu.synchronize()
            r = gqa_official(inp, c)
            torch.npu.synchronize()
            acc = accuracy(s, r)
            s_ms = bench(gqa_sink, inp, c, iters=iters)
            o_ms = bench(gqa_official, inp, c, iters=iters)
        except Exception as e:
            acc, s_ms, o_ms = {"bit": None, "mere": None, "max_abs": None,
                               "note": f"fail: {str(e)[:60]}"}, None, None
        ratio = s_ms / o_ms if (s_ms and o_ms) else None
        results["GQA_" + name] = {"acc": acc, "sink_ms": s_ms, "official_ms": o_ms, "ratio": ratio}
        m = acc["mere"]
        print(f"{'GQA_' + name:<22}{str(acc['bit']):>6}{(f'{m:.2e}' if m is not None else 'N/A'):>11}"
              f"{(f'{s_ms:.3f}' if s_ms else 'N/A'):>10}"
              f"{(f'{o_ms:.3f}' if o_ms else 'N/A'):>9}"
              f"{(f'{ratio:.3f}' if ratio else 'N/A'):>8}")
        try:
            torch.npu.empty_cache()
        except Exception:
            pass

    if args.only:
        key = args.only if args.only in results else "GQA_" + args.only
        rec = results.get(key, {"error": "no result (crash/timeout)"})
        print("RESULT:" + json.dumps({"case": key, **rec}, ensure_ascii=False))
        return

    with open(args.json, "w") as f:
        json.dump({"device": "Ascend950PR card2", "scope": "full family acc+perf",
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.json}")


if __name__ == "__main__":
    main()
