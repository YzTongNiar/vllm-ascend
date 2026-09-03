"""kw-6 探针判读：ProcessVec1Vf dst 布局测定（m=2, N=64, scale=1）。

输入模式：row0 x_c=0.01c → exp0_c = exp(0.01(c-63))；row1 x_c=0.01c+0.005 → exp1_c = exp(0.01(c-63)+0.005)
（max over 64 cols = row 的 x_63；两行值恒差 exp 因子 e^0.005≈1.005）
候选布局的预测（dst bf16 偏移，元素单位）：
  A) k-块步长 33（vec1Srcstride 元素制）:   block j 数据在 [33j, 33j+?]
  B) k-块步长 64（s1BaseSize 元素制）:      block j 在 [64j]
  C) 16x16 分形 NZ（块 256 元素）:          block j 在 [256j]，行 r 在 [256j + 16r]
  D) k-块步长 33*16=528（33 以 32B 块计）:  block j 在 [528j]
dump 范围 864 bf16 → A/B/C 块 0-2 可见、D 块 0-1 可见。
"""
import glob, importlib.machinery, importlib.util, math, os, struct, torch, torch_npu

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l)
_e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)

T, N1, N2, D, BS, KV = 7, 8, 8, 192, 128, 20
pool = math.ceil(KV / BS) + 2
g = torch.Generator().manual_seed(21)
kp = (torch.rand(pool, N2, BS, D, generator=g) * 2).bfloat16()
bt = torch.arange(pool, dtype=torch.int32).view(1, -1)
q0 = torch.zeros(T, N1, D).bfloat16()
meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
    N1, N2, D, D,
    actual_seq_lengths=torch.tensor([T], dtype=torch.int64, device="npu"),
    actual_seq_lengths_kv=torch.tensor([KV], dtype=torch.int64, device="npu"),
    input_layout="TND", input_layout_kv="PA", sparse_mode=0, block_size=BS, rope_head_dim=0)
out, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    q0.npu(), kp.npu(), kp.npu(),
    actual_seq_qlen=torch.tensor([T], dtype=torch.int64, device="npu"),
    actual_seq_kvlen=torch.tensor([KV], dtype=torch.int64, device="npu"),
    block_table=bt.npu(), meta_data=meta,
    num_query_heads=N1, num_key_value_heads=N2, softmax_scale=1.0 / math.sqrt(D),
    input_layout="TND", sparse_mode=0, block_size=BS, return_softmax_lse=False)
torch.npu.synchronize()
raw = meta.cpu().numpy().tobytes()
i16 = struct.unpack("<%dh" % (len(raw) // 2), raw)
dst = i16[1184:1184 + 864]  # bf16 位样


def bf16_bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


f = [bf16_bits_to_float(b) for b in dst]
print("dst[0..15] float:", [f"{v:.4f}" for v in f[:16]])
print("dst[16..33]:", [f"{v:.4f}" for v in f[16:33]])
print("dst[33..49]:", [f"{v:.4f}" for v in f[33:49]])
print("dst[64..80]:", [f"{v:.4f}" for v in f[64:80]])
print("dst[128..144]:", [f"{v:.4f}" for v in f[128:144]])
print("dst[256..272]:", [f"{v:.4f}" for v in f[256:272]])
print("dst[528..544]:", [f"{v:.4f}" for v in f[528:544]])
exp0 = [math.exp(0.01 * (c - 63)) for c in range(64)]
exp1 = [math.exp(0.01 * (c - 63) + 0.005) for c in range(64)]
print("expect exp0[0:4]:", [f"{v:.4f}" for v in exp0[:4]], " exp0[16:20]:", [f"{v:.4f}" for v in exp0[16:20]])
print("expect exp1[0:4]:", [f"{v:.4f}" for v in exp1[:4]])
# 自动匹配：在 dst 里找 exp0 序列的起点
for name, seq in (("row0", exp0), ("row1", exp1)):
    hits = [i for i in range(0, 800) if abs(f[i] - seq[0]) < 2e-3 and abs(f[i + 1] - seq[1]) < 2e-3
            and abs(f[i + 2] - seq[2]) < 2e-3 and abs(f[i + 3] - seq[3]) < 2e-3]
    print(f"{name} 序列起点候选: {hits}")
