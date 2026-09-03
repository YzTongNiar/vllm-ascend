"""kw-3 第一刀判读：纯探针版（原 kernel 路径 + A'/A''/E/D' 四探针）。

判据（coordinator kw-3 指令）：
  D' 随 q 变化(两组输入读数不同) => cube Q 路径本来就对，病灶在 vector 侧
  D' 与 q 无关(两组读数逐位同)  => 病灶在 cube Q→L0A 装载（原 load3d 在 c310 失效）
  E == 64×1.0                   => vector 侧 plain VEC 计算能力正常
  E 脏                          => vector 侧 VEC 层失效
  A'/A'' 差异                   => Muls/ElewiseCompute 是否污染 P
"""
import glob, importlib.machinery, importlib.util, math, os, sys, torch, torch_npu
PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l); _e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)
T, N1, N2, D, BS, KVLEN = 7, 8, 8, 192, 128, 20
pool = math.ceil(KVLEN / BS) + 2
g = torch.Generator().manual_seed(21)
kp = (torch.rand(pool, N2, BS, D, generator=g) * 2).bfloat16()
bt = torch.arange(pool, dtype=torch.int32).view(1, -1)
def run(qval):
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
        num_query_heads=N1, num_key_value_heads=N2, softmax_scale=1.0/math.sqrt(D),
        input_layout="TND", sparse_mode=0, block_size=BS, return_softmax_lse=True)
    torch.npu.synchronize()
    return meta.float().cpu(), out, lse
m0, o0, l0 = run(0)
m8, o8, l8 = run(8)
def reg(tag, mf, name, lo, n):
    r = mf[lo:lo+n]
    print(f"[{tag} {name}] nz={int((r!=0).sum())}/{n} max={r.abs().max().item():.6g} h8={[f'{v:.5g}' for v in r[:8].tolist()]}")
    return r
for tag, mf in (("q=0", m0), ("q=8", m8)):
    reg(tag, mf, "A' preMuls ", 592, 64)
    reg(tag, mf, "A'' postMuls", 824, 64)
    reg(tag, mf, "E VECself  ", 888, 64)
    reg(tag, mf, "D' L0C@fixp", 960, 64)
print(f"[D'-q-invariance] D'(q=0) == D'(q=8) 逐位: {torch.equal(m0[960:1024], m8[960:1024])}")
print(f"[out] q=0 head4={[f'{v:.4g}' for v in o0.reshape(-1)[:4].tolist()]}  q=8 head4={[f'{v:.4g}' for v in o8.reshape(-1)[:4].tolist()]}")
