"""kw-8 C04 快检：MLA 最小单块 case 精度。"""
import glob, importlib.machinery, importlib.util, math, os, sys, torch, torch_npu
sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l); _e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)
SCALE = 0.0721687836487032
def run_case(tag, H, kvh, D, R, BS, B, T, kvs, qls, seed, sparse=0, use_mask=False, mask_rc=None):
    import case_builder as cb
    c = dict(case_name=tag, scenario="kw8", B=B, G=0, H=H, D=D, R=R, BS=BS, T=sum(qls),
             q_len=qls, kvlen=kvs, pool=0, bt_w=max(math.ceil(k/BS) for k in kvs), use_mask=use_mask,
             mask_rows=mask_rc[0] if mask_rc else 0, mask_cols=mask_rc[1] if mask_rc else 0,
             sparse_mode=sparse, seed=seed)
    # 走本地 build_inputs 逻辑（case_builder.build_inputs 需要 G/T 字段齐全）
    g = torch.Generator().manual_seed(seed)
    Tt = c["T"]
    q = (torch.rand(Tt, H, D, generator=g)*2).bfloat16()
    qpe = (torch.rand(Tt, H, R, generator=g)*2).bfloat16()
    pool = sum(math.ceil(L/BS) for L in kvs)+2
    kpool = (torch.rand(pool, kvh, BS, D, generator=g)*2).bfloat16()
    krpool = (torch.rand(pool, kvh, BS, R, generator=g)*2).bfloat16()
    bt = torch.zeros(B, c["bt_w"], dtype=torch.int32); st=0
    for i,L in enumerate(kvs):
        nb = math.ceil(L/BS); bt[i,:nb]=torch.arange(st,st+nb); st+=nb
    qcum=[]; acc=0
    for x in qls: acc+=x; qcum.append(acc)
    q, qpe, kpool, krpool, bt = q.npu(), qpe.npu(), kpool.npu(), krpool.npu(), bt.npu()
    ql = torch.tensor(qcum, dtype=torch.int64, device="npu"); kl = torch.tensor(kvs, dtype=torch.int64, device="npu")
    mask = None
    if use_mask:
        mask = torch.triu(torch.ones(mask_rc[0], mask_rc[1], dtype=torch.int8), diagonal=1).npu()
    ob, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q, kpool, kpool, query_rope=qpe, key_rope=krpool, atten_mask=mask,
        num_query_heads=H, num_key_value_heads=kvh, input_layout="TND", sparse_mode=sparse,
        softmax_scale=SCALE, block_table=bt, block_size=BS, actual_seq_qlen=qcum,
        actual_seq_kvlen=list(kvs), return_softmax_lse=False)
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        H, kvh, D, D, actual_seq_lengths=ql, actual_seq_lengths_kv=kl, batch_size=B,
        sparse_mode=sparse, input_layout="TND", input_layout_kv="PA", rope_head_dim=R,
        block_size=BS, aic_core_num=28, aiv_core_num=56)
    kw = dict(query_rope=qpe, key_rope=krpool)
    if mask is not None: kw["atten_mask"] = mask
    on_, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        q, kpool, kpool, actual_seq_qlen=ql, actual_seq_kvlen=kl, block_table=bt,
        meta_data=meta, num_query_heads=H, num_key_value_heads=kvH if (kvH:=kvh) else kvh,
        softmax_scale=SCALE, input_layout="TND", sparse_mode=sparse, block_size=BS, **kw)
    torch.npu.synchronize()
    d = (on_.float()-ob.float()).abs()
    mere = (d/ob.float().abs().clamp_min(1e-6)).mean().item()
    print(f"[{tag}] bit={torch.equal(on_.cpu(), ob.cpu())} mere={mere:.4g} max={d.max().item():.4g}")
# C04: B1 T7 H8 kvh1 D512 R64 kv128 sm0
run_case("C04_sig_b_min", 8, 1, 512, 64, 128, 1, 7, [128], [7], 20260824)
# C05: B4 T8each H16 kvh12 D512 R64 kv4096+ sm3 mask
run_case("C05_batch4", 12, 1, 512, 64, 128, 4, 32, [4096,4160,4224,4288], [8,8,8,8], 20260825, sparse=3, use_mask=True, mask_rc=(2048,2048))
# C12: r12 kv128k B1 T8 H16 kvh12
run_case("C12_r12_kv128k", 12, 1, 512, 64, 128, 1, 8, [131072], [8], 20260901, sparse=3, use_mask=True, mask_rc=(2048,2048))
