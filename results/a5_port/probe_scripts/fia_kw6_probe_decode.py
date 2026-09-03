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
meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
    N1, N2, D, D, actual_seq_lengths=torch.tensor([T],dtype=torch.int64,device="npu"),
    actual_seq_lengths_kv=torch.tensor([KV],dtype=torch.int64,device="npu"),
    input_layout="TND", input_layout_kv="PA", sparse_mode=0, block_size=BS, rope_head_dim=0)
out, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    q0.npu(), kp.npu(), kp.npu(), actual_seq_qlen=torch.tensor([T],dtype=torch.int64,device="npu"),
    actual_seq_kvlen=torch.tensor([KV],dtype=torch.int64,device="npu"), block_table=bt.npu(), meta_data=meta,
    num_query_heads=N1, num_key_value_heads=N2, softmax_scale=1.0/math.sqrt(D),
    input_layout="TND", sparse_mode=0, block_size=BS, return_softmax_lse=False)
torch.npu.synchronize()
raw = meta.cpu().numpy().tobytes()
i16 = struct.unpack("<%dh" % (len(raw)//2), raw)
dst = i16[1184:1184+640]; src = i16[1824:1824+128]; slot = i16[1952:1952+48]
def bf16(b):
    return struct.unpack("<f", struct.pack("<I", (b & 0xFFFF) << 16))[0]
f = [bf16(b) for b in dst]
print("dst[0:8]  :", [f"{v:.4f}" for v in f[:8]], " hex:", [hex(dst[i]&0xFFFF) for i in range(4)])
print("dst[16:24]:", [f"{v:.4f}" for v in f[16:24]])
print("dst[33:41]:", [f"{v:.4f}" for v in f[33:41]])
print("dst[64:72]:", [f"{v:.4f}" for v in f[64:72]])
print("dst[128:136]:", [f"{v:.4f}" for v in f[128:136]])
print("dst[256:264]:", [f"{v:.4f}" for v in f[256:264]])
# src 回读: float 位样两半
sf = [struct.unpack("<f", struct.pack("<hH", src[2*i], src[2*i+1] & 0xFFFF))[0] for i in range(16)]
print("src[0:8]  :", [f"{v:.4f}" for v in sf[:8]])
slotf = [struct.unpack("<f", struct.pack("<hH", slot[2*i], slot[2*i+1] & 0xFFFF))[0] for i in range(6)]
print("slots(sum,max 前3):", [f"{v:.4g}" for v in slotf[:6]])
exp0 = [math.exp(0.01*(c-63)) for c in range(64)]
for name, seq in (("row0", exp0),):
    hits = [i for i in range(0, 636) if abs(f[i]-seq[0])<3e-3 and abs(f[i+1]-seq[1])<3e-3 and abs(f[i+2]-seq[2])<3e-3]
    print(f"{name} 序列起点:", hits)
