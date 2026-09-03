import glob, importlib.machinery, importlib.util, math, os, torch, torch_npu
PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l); _e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)
def run_case(tag, T, N1, N2, D, BS, kvlen, B=1):
    pool = (math.ceil(kvlen/BS)+2)*B
    g = torch.Generator().manual_seed(21)
    kp = (torch.rand(pool, N2, BS, D, generator=g)*2).bfloat16()
    vp = (torch.rand(pool, N2, BS, D, generator=g)*2).bfloat16()
    q = torch.zeros(T, N1, D).bfloat16()
    nb = math.ceil(kvlen/BS)
    bt = torch.zeros(B, nb, dtype=torch.int32)
    start = 0
    for i in range(B):
        bt[i,:]=torch.arange(start,start+nb); start+=nb
    ql = torch.tensor([T//B]*B, dtype=torch.int64)
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
    o = out.float().cpu()  # [T, N1, D]
    # 正确参照：head h 的 V 均值（g=0 时所有 q 头同权重；MHA N1==N2 时头一一对应；
    # GQA 时 q 头 j 属 kv 头 h=j//gS —— 本例 MHA）
    ref = torch.zeros(B, N2, D)
    for i in range(B):
        cnt = 0
        for j in range(nb):
            used = min(BS, kvlen-cnt)
            ref[i] += vp[bt[i,j].item()].float().sum(dim=1)[: , :used].sum(dim=1) if used==BS else vp[bt[i,j].item()][:, :used].float().sum(dim=(0,1))
            cnt += used
        ref[i] /= kvlen
    d = (o.view(B, -1, N1, D)[:, :, :] - ref[:, None, :, :]).abs() if N1==N2 else (o.view(B,-1,N1,D)-ref[:,None,:N2,:].expand(B,1,N2,D) if N1==N2 else (o.view(B,-1,N1,D)))
    # 简化: MHA
    dd = (o.view(B, T//B, N1, D) - ref[:, None, :, :]).abs()
    print(f"[{tag}] q=0 out-vs-Vmean maxabs={dd.max().item():.6f} mean={dd.mean().item():.6f}")
run_case("kv20   T7H8D192 1loop", 7, 8, 8, 192, 128, 20)
run_case("kv248  T14H8D192 1loop", 14, 8, 8, 192, 128, 248, B=2)
run_case("kv4096 T7H8D192 8loop", 7, 8, 8, 192, 128, 4096)
run_case("kv32768 T7H8D192 64loop", 7, 8, 8, 192, 128, 32768)
