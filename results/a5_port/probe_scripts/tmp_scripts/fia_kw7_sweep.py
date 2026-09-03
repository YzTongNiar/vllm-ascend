"""kw-7 多窗全量 dump：19 窗拼出 tmpBuff1 全 32KB，定位 VF store 布局。"""
import glob, importlib.machinery, importlib.util, math, os, struct, torch, torch_npu
PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l); _e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)
T, N1, N2, D, BS, KV = 7, 8, 8, 192, 128, 20
pool = math.ceil(KV/BS)+2
g = torch.Generator().manual_seed(21)
kp = (torch.rand(pool, N2, BS, D, generator=g)*2).bfloat16()
bt = torch.arange(pool, dtype=torch.int32).view(1,-1)
q0 = torch.zeros(T, N1, D).bfloat16()
full = []
for win in range(19):
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=torch.tensor([T],dtype=torch.int64,device="npu"),
        actual_seq_lengths_kv=torch.tensor([KV],dtype=torch.int64,device="npu"),
        input_layout="TND", input_layout_kv="PA", sparse_mode=0, block_size=BS, rope_head_dim=0)
    meta[600] = win  # 窗口选择器
    out, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        q0.npu(), kp.npu(), kp.npu(), actual_seq_qlen=torch.tensor([T],dtype=torch.int64,device="npu"),
        actual_seq_kvlen=torch.tensor([KV],dtype=torch.int64,device="npu"), block_table=bt.npu(), meta_data=meta,
        num_query_heads=N1, num_key_value_heads=N2, softmax_scale=1.0/math.sqrt(D),
        input_layout="TND", sparse_mode=0, block_size=BS, return_softmax_lse=False)
    torch.npu.synchronize()
    raw = meta.cpu().numpy().tobytes()
    i16 = struct.unpack("<%dh" % (len(raw)//2), raw)
    full.extend(i16[1184:1184+864])
    print(f"win{win} ok", flush=True)
print("total int16:", len(full))
def bf16(b):
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]
# 1) 找 exp0 序列（0.5326, 0.5379, ...）
exp0 = [math.exp(0.01*(c-63)) for c in range(64)]
exp1 = [math.exp(0.01*(c-63)+0.005) for c in range(64)]
def find(seq):
    hits = []
    for i in range(len(full)-8):
        if all(abs(bf16(full[i+k]) - seq[k]) < 3e-3 for k in range(8)):
            hits.append(i)
    return hits
h0, h1 = find(exp0), find(exp1)
print("row0 exp 序列起点(int16):", h0)
print("row1 exp 序列起点(int16):", h1)
# 2) 找魔数区边界
magic_pos = [i for i, b in enumerate(full) if (b & 0xFFFF) == 0x3F3D]
if magic_pos:
    print(f"魔数区: [{min(magic_pos)}, {max(magic_pos)}] 共 {len(magic_pos)}")
nz = [i for i, b in enumerate(full) if b != 0 and (b & 0xFFFF) != 0x3F3D]
print(f"非零非魔数位个数: {len(nz)} 前 12 位: {nz[:12]}")
# 3) 打印关键区
for base in (0, 256, 512, 1024, 2048, 4096):
    print(f"int16[{base}..{base+12}]:", [f"{bf16(b):.4f}" if 0 < abs(bf16(b)) < 1e37 else hex(b & 0xFFFF) for b in full[base:base+12]])
import json
open("/tmp/fia_kw7_full_dump.json","w").write(json.dumps(full))
print("dump saved /tmp/fia_kw7_full_dump.json")
