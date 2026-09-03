"""FIA v2 Sink（A5）vs 官方算子 双方法性能对照 — wall-clock 口径。

用法（等 kernel 修复、六门禁全绿后执行）：
  cd /home/t00886357/a5_port_base_supplement/tests_pr15336
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=2
  export FIA_GATE_TREE=/home/t00886357/fia_sink_a5_port_base/vllm-ascend
  OPP=$FIA_GATE_TREE/vllm_ascend/_cann_ops_custom
  export ASCEND_CUSTOM_OPP_PATH=$OPP:$OPP/vendors/custom_transformer
  python3 /home/t00886357/fia_sink_a5_port_base/results/a5_port/perf/perf_sink_vs_official.py \
      [--json out.json] [--iters 20]

口径：
- sink 侧 = 两段式完整成本（metadata 前置 + 主算子），X47 成本判据口径
- 官方侧 = npu_fused_infer_attention_score_v2（host list 传参）
- warmup 3 + 计时 iters 轮（逐轮 synchronize），报均值/中位
- 用例：S1/S2（kernel-bound 长序列）、kvlen_4096（中间档）、g1_short（launch-bound）
"""
import argparse
import glob
import importlib.machinery
import importlib.util
import json
import math
import os
import time

import torch
import torch_npu

HERE = os.path.dirname(os.path.abspath(__file__))


def load_ext(pr_tree):
    c = sorted(glob.glob(os.path.join(pr_tree, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
    ldr = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", c)
    spec = importlib.util.spec_from_loader("vllm_ascend_C", ldr)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_case(name):
    """返回 (inputs, case_dict)。与门禁 G1/S1/S2 同构构造。"""
    torch.manual_seed(99)
    cases = {
        # X47 成本判据 kernel-bound 主用例（对位 builder S1/S2）
        "S1_r96_gs8_kv256k": dict(T=8, N1=96, N2=12, D=128, kvlen=262144),
        "S2_r96_gs8_kv512k": dict(T=8, N1=96, N2=12, D=128, kvlen=524288),
        "kvlen_4096":        dict(T=7, N1=8, N2=8, D=192, kvlen=4096),
        "g1_short":          dict(T=7, N1=8, N2=8, D=192, kvlen=20),
    }
    c = cases[name]
    BS = 128
    B, T = 1, c["T"]
    pool = math.ceil(c["kvlen"] / BS) + 2
    q = (torch.rand(T, c["N1"], c["D"]) * 2).bfloat16().npu()
    kp = (torch.rand(pool, c["N2"], BS, c["D"]) * 2).bfloat16().npu()
    vp = (torch.rand(pool, c["N2"], BS, c["D"]) * 2).bfloat16().npu()
    bt = torch.arange(pool, dtype=torch.int32).view(B, -1).npu()
    qd = torch.tensor([T], dtype=torch.int64, device="npu")
    kd = torch.tensor([c["kvlen"]], dtype=torch.int64, device="npu")
    return dict(q=q, kp=kp, vp=vp, bt=bt, qd=qd, kd=kd, scale=1.0 / math.sqrt(c["D"]),
                N1=c["N1"], N2=c["N2"], D=c["D"], BS=BS, T=T, kvlen=c["kvlen"])


def call_sink(inp):
    """两段式完整成本（每轮新 metadata，eager 正常用法）。"""
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        inp["N1"], inp["N2"], inp["D"], inp["D"],
        actual_seq_lengths=inp["qd"].clone(), actual_seq_lengths_kv=inp["kd"].clone(),
        batch_size=1, sparse_mode=0, input_layout="TND", input_layout_kv="PA",
        rope_head_dim=0, block_size=inp["BS"],
        aic_core_num=28, aiv_core_num=56)
    out, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kp"], inp["vp"], actual_seq_qlen=inp["qd"],
        actual_seq_kvlen=inp["kd"], block_table=inp["bt"], meta_data=meta,
        num_query_heads=inp["N1"], num_key_value_heads=inp["N2"],
        softmax_scale=inp["scale"], input_layout="TND", sparse_mode=0,
        block_size=inp["BS"])
    return out


def call_official(inp):
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        inp["q"], inp["kp"], inp["vp"], num_query_heads=inp["N1"],
        num_key_value_heads=inp["N2"], input_layout="TND", sparse_mode=0,
        softmax_scale=inp["scale"], block_table=inp["bt"], block_size=inp["BS"],
        actual_seq_qlen=[inp["T"]], actual_seq_kvlen=[inp["kvlen"]],
        return_softmax_lse=False)
    return out


def bench(fn, inp, warmup=3, iters=20):
    for _ in range(warmup):
        fn(inp)
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(inp)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    ts.sort()
    return {"mean_ms": sum(ts) / len(ts), "median_ms": ts[len(ts) // 2],
            "min_ms": ts[0], "max_ms": ts[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "perf_a5_result.json"))
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    pr_tree = os.environ.get(
        "FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
    load_ext(pr_tree)

    results = {}
    print(f"{'case':<20}{'sink med(ms)':>14}{'official med':>14}{'ratio':>9}")
    for name in ["S1_r96_gs8_kv256k", "S2_r96_gs8_kv512k", "kvlen_4096", "g1_short"]:
        inp = build_case(name)
        # 正确性快检（bit）——性能测试前置：不 bit 一致则标记
        eq = bool(torch.equal(call_sink(inp).cpu(), call_official(inp).cpu()))
        s = bench(call_sink, inp, iters=args.iters)
        o = bench(call_official, inp, iters=args.iters)
        ratio = s["median_ms"] / o["median_ms"] if o["median_ms"] > 0 else float("nan")
        results[name] = {"sink": s, "official": o, "ratio_vs_official": ratio,
                         "bit_exact": eq}
        print(f"{name:<20}{s['median_ms']:>14.3f}{o['median_ms']:>14.3f}{ratio:>9.3f}"
              f"  bit={eq}")

    with open(args.json, "w") as f:
        json.dump({"device": "Ascend950PR card2", "scope": "two-stage sink vs official",
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.json}")


if __name__ == "__main__":
    main()
