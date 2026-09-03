import glob, importlib.machinery, importlib.util, math, os, torch, torch_npu
PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l); _e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)
T, N1, N2, D, BS = 9, 16, 8, 192, 128
def iso(kv):
    pool = math.ceil(kv/BS)+2
    g = torch.Generator().manual_seed(33)
    kp = (torch.rand(pool, N2, BS, D, generator=g)*2).bfloat16()
    vp = (torch.rand(pool, N2, BS, D, generator=g)*2).bfloat16()
    q = torch.zeros(T, N1, D).bfloat16()
    bt = torch.arange(pool, dtype=torch.int32).view(1,-1)
    ob, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q.npu(), kp.npu(), vp.npu(), num_query_heads=N1, num_key_value_heads=N2,
        input_layout="TND", sparse_mode=0, softmax_scale=1.0/math.sqrt(D), block_table=bt.npu(),
        block_size=BS, actual_seq_qlen=[T], actual_seq_kvlen=[kv], return_softmax_lse=False)
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=torch.tensor([T],dtype=torch.int64,device="npu"),
        actual_seq_lengths_kv=torch.tensor([kv],dtype=torch.int64,device="npu"),
        input_layout="TND", input_layout_kv="PA", sparse_mode=0, block_size=BS, rope_head_dim=0)
    on_, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        q.npu(), kp.npu(), vp.npu(), actual_seq_qlen=torch.tensor([T],dtype=torch.int64,device="npu"),
        actual_seq_kvlen=torch.tensor([kv],dtype=torch.int64,device="npu"), block_table=bt.npu(),
        meta_data=meta, num_query_heads=N1, num_key_value_heads=N2, softmax_scale=1.0/math.sqrt(D),
        input_layout="TND", sparse_mode=0, block_size=BS, return_softmax_lse=False)
    torch.npu.synchronize()
    d = (on_.float()-ob.float()).abs()
    print(f"[kv={kv:5d}] bit={torch.equal(on_.cpu(), ob.cpu())} maxabs={d.max().item():.4g}")
for kv in (16, 77, 128, 129, 130, 131, 160, 192, 248, 256):
    iso(kv)
