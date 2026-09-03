import glob, importlib.machinery, importlib.util, math, os, sys, torch, torch_npu
PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l); _e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)
def run_case(tag, T, N1, N2, D, BS, kvlen, B=1, qmode="zero"):
    pool = (math.ceil(kvlen/BS)+2) * B
    g = torch.Generator().manual_seed(21)
    kp = (torch.rand(pool, N2, BS, D, generator=g)*2).bfloat16()
    vp = (torch.rand(pool, N2, BS, D, generator=g)*2).bfloat16()
    q = torch.zeros(T, N1, D).bfloat16()
    bt = torch.zeros(B, math.ceil(kvlen/BS), dtype=torch.int32); start=0
    for i in range(B):
        nb = math.ceil(kvlen/BS); bt[i,:nb]=torch.arange(start,start+nb); start+=nb
    ql = torch.tensor([T]*B, dtype=torch.int64) if B==1 else torch.tensor([T//B]*B, dtype=torch.int64)
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=ql.npu(),
        actual_seq_lengths_kv=torch.full((B,), kvlen, dtype=torch.int64).npu(),
        input_layout="TND", input_layout_kv="PA", sparse_mode=0, block_size=BS, rope_head_dim=0)
    out, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        q.npu(), kp.npu(), vp.npu(), actual_seq_qlen=ql.npu(),
        actual_seq_kvlen=torch.full((B,), kvlen, dtype=torch.int64).npu(),
        block_table=bt.npu(), meta_data=meta, num_query_heads=N1, num_key_value_heads=N2,
        softmax_scale=1.0/math.sqrt(D), input_layout="TND", sparse_mode=0, block_size=BS,
        return_softmax_lse=False)
    torch.npu.synchronize()
    # q=0: out 应=每头 V 均值（该头 kvlen 个 token 的 D 维均值）
    ok = True  # vmean unused
    if False: vmean = torch.stack([vp[bt[0,j].item(),:,:,:].float().mean(dim=(0,1)) for j in range(min(math.ceil(kvlen/BS), bt.shape[1]))])
    # 简化：全池均值近似仅当单 batch 全用; 精确按 head:
    ok = True
    vm = torch.zeros(N1, D)
    for j in range(math.ceil(kvlen/BS)):
        vm += vp[bt[0,j].item()].float().sum(dim=(0,1))
    vm /= kvlen
    d = (out[0].float().cpu() - vm).abs()
    print(f"[{tag}] out-vs-Vmean maxabs={d.max().item():.6f} mean={d.mean().item():.6f} (q=0 受控, 应≈0)")
run_case("kv4096  T7H8D192", 7, 8, 8, 192, 128, 4096)
run_case("kv32768 T7H8D192", 7, 8, 8, 192, 128, 32768)
run_case("b2_248  T14H8D192", 14, 8, 8, 192, 128, 248, B=2)
