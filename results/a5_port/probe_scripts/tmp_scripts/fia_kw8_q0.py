"""kw-8 零成本快检：C04 形态（B1 kv128 单块 SigB 无 mask）q=0 受控 + 差异模式。"""
import glob, importlib.machinery, importlib.util, math, os, sys, torch, torch_npu
sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l); _e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)
SCALE = 0.0721687836487032
H, KVH, D, R, BS = 8, 1, 512, 64, 128
B, T, KV = 1, 7, 128
pool, bt_w = 2, 1
def run(qv, seed=20260824):
    g = torch.Generator().manual_seed(seed)
    qpe = (torch.rand(T, H, R, generator=g)*2).bfloat16()
    kpool = (torch.rand(pool, 1, BS, D, generator=g)*2).bfloat16()
    krpool = (torch.rand(pool, 1, BS, R, generator=g)*2).bfloat16()
    q = torch.zeros(T, H, D).bfloat16() if qv == 0 else torch.full((T, H, D), float(qv)).bfloat16()
    if qv != 0: qpe = torch.zeros_like(qpe)   # rope 侧也置 0/常值，保证 P 恒定
    bt = torch.arange(pool, dtype=torch.int32).view(1, -1)
    ql = torch.tensor([T], dtype=torch.int64); kl = torch.tensor([KV], dtype=torch.int64)
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        H, KVH, D, D, actual_seq_lengths=ql.npu(), actual_seq_lengths_kv=kl.npu(),
        batch_size=B, sparse_mode=0, input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)
    out, lse = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        q.npu(), kpool.npu(), kpool.npu(), query_rope=qpe.npu(), key_rope=krpool.npu(),
        actual_seq_qlen=ql.npu(), actual_seq_kvlen=kl.npu(), block_table=bt.npu(), meta_data=meta,
        num_query_heads=H, num_key_value_heads=KVH, softmax_scale=SCALE, input_layout="TND",
        sparse_mode=0, block_size=BS, return_softmax_lse=True)
    ob, lseb = torch_npu.npu_fused_infer_attention_score_v2(
        q.npu(), kpool.npu(), kpool.npu(), query_rope=qpe.npu(), key_rope=krpool.npu(),
        num_query_heads=H, num_key_value_heads=KVH, input_layout="TND", sparse_mode=0,
        softmax_scale=SCALE, block_table=bt.npu(), block_size=BS, actual_seq_qlen=[T],
        actual_seq_kvlen=[KV], return_softmax_lse=True)
    torch.npu.synchronize()
    return out, lse, ob, lseb, kpool, krpool
out0, lse0, ob0, lseb0, kp, krp = run(0)
print(f"[q=0] lse 应全 log(128)={math.log(128):.4f}: 我方 head4={[f'{v:.3g}' for v in lse0.float().cpu().reshape(-1)[:4].tolist()]}")
print(f"[q=0] lse 官方 head4={[f'{v:.3g}' for v in lseb0.float().cpu().reshape(-1)[:4].tolist()]}  正确数={int(((lse0.float().cpu().reshape(-1)-math.log(128)).abs()<1e-3).sum())}/56")
d = (out0.float().cpu()-ob0.float().cpu()).abs()
print(f"[q=0] out-vs-官方 max={d.max().item():.4g}  out 我方 head2={out0.float().cpu().reshape(-1)[:2].tolist()} 官方={ob0.float().cpu().reshape(-1)[:2].tolist()}")
# V 均值参照（q=0 → out=V 前 kv 个 token 均值, 含 rope 无贡献）
vm = kp[0, 0, :KV].float().mean(dim=0)  # [D]
dd = (out0.float().cpu()[0,0]-vm).abs()
print(f"[q=0] out[0,0] vs V均值: max={dd.max().item():.4g}")
out8, lse8, ob8, _, _, _ = run(8)
d8 = (out8.float().cpu()-ob8.float().cpu()).abs()
print(f"[q=8] out-vs-官方 max={d8.max().item():.4g}")
