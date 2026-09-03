"""Round-5 判读：metadata 通道 dump（不被输出覆写）。

metadata int32[1024]，AIC/AIV/base 表占 [0..585]，尾区 dump：
  A' [592..815]  fp32: P 到达 vector 侧的快照（Vec1 读后、softmax 前，首块）
  B' [824..887]  fp32: Vec2 阶段回读 mm1Res GM slot 前 64（终态参考）
  D' [896..959]  fp32: AIC 首块 L0C 经生产同款 fixpipe 直写（2行×dstStride32）

q=0（P 应恒 0）判读：
  D' 全 0 + A' 全 0      => P 生成与传递都正确，病灶在 softmax/vec 内部（下一步 vec 侧）
  D' 全 0 + A' 脏        => L0C 正确但 GM slot 脏 => fixpipe 写出或跨核同步（写侧/读侧）
  D' 脏                  => mm1/L0C 生成错（Q 装载/mmad），病灶在 cube 内部
"""
import glob
import importlib.machinery
import importlib.util
import math
import os

import torch
import torch_npu

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_cand = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_ldr = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _cand[-1])
_spec = importlib.util.spec_from_loader("vllm_ascend_C", _ldr)
_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ext)

T, N1, N2, D, BS = 7, 8, 8, 192, 128
KVLEN = int(os.environ.get("DUMP_KVLEN", "20"))


def run(qval):
    pool = math.ceil(KVLEN / BS) + 2
    g = torch.Generator().manual_seed(21)
    kp = (torch.rand(pool, N2, BS, D, generator=g) * 2).bfloat16()
    bt = torch.arange(pool, dtype=torch.int32).view(1, -1)
    q = torch.zeros(T, N1, D).bfloat16() if qval == 0 else torch.full((T, N1, D), float(qval)).bfloat16()
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D,
        actual_seq_lengths=torch.tensor([T], dtype=torch.int64, device="npu"),
        actual_seq_lengths_kv=torch.tensor([KVLEN], dtype=torch.int64, device="npu"),
        input_layout="TND", input_layout_kv="PA", sparse_mode=0, block_size=BS, rope_head_dim=0)
    out, lse = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        q.npu(), kp.npu(), kp.npu(),
        actual_seq_qlen=torch.tensor([T], dtype=torch.int64, device="npu"),
        actual_seq_kvlen=torch.tensor([KVLEN], dtype=torch.int64, device="npu"),
        block_table=bt.npu(), meta_data=meta,
        num_query_heads=N1, num_key_value_heads=N2, softmax_scale=1.0 / math.sqrt(D),
        input_layout="TND", sparse_mode=0, block_size=BS, return_softmax_lse=True)
    torch.npu.synchronize()
    return meta, out, lse, kp


def region(mf, name, lo, length):
    r = mf[lo:lo + length]
    print(f"[{name}] int32[{lo}..] nonzero={int((r != 0).sum())}/{r.numel()} "
          f"maxabs={r.abs().max().item():.6g} head8={[f'{v:.4g}' for v in r[:8].tolist()]}")
    return r


for qv in (0, 8):
    meta, out, lse, kp = run(qv)
    mf = meta.float().cpu()  # int32 视图转 float 打印（位样直读）
    print(f"===== q={qv} kvlen={KVLEN} =====")
    region(mf, "A'(P@Vec1)", 592, 224)
    region(mf, "B'(slot@Vec2)", 824, 64)
    region(mf, "D'(L0C@fixpipe)", 896, 64)
    lf = lse.float().cpu().reshape(-1)
    print(f"[lse] head4={[f'{v:.4g}' for v in lf[:4].tolist()]} expect(log{KVLEN})={math.log(KVLEN):.4f}")

# 全量 lse 模式分析（q=0 期望全 log(20)=2.9957）
meta, out, lse, kp = run(0)
lf = lse.float().cpu().reshape(T, N1)
ok = (lf - math.log(KVLEN)).abs() < 1e-3
print("===== q=0 lse 全量 (t,h) 正确性矩阵 =====")
for t in range(T):
    print(f"t={t}: " + " ".join("OK " if ok[t, h] else "XX " for h in range(N1)) +
          f" vals={[f'{v:.3g}' for v in lf[t, :3].tolist()]}")
print(f"n_correct={int(ok.sum())}/{T*N1}")
vm = kp[0, :, :KVLEN].float().mean(dim=1).unsqueeze(0).expand(T, N1, D)
o = out.float().cpu()
d = (o - vm).abs()
print(f"[out-vs-Vmean] maxabs={d.max().item():.4f} mean={d.mean().item():.4f}")
