"""kw-9: C05/C10 脏 batch 隔离实验。
1) C10 三跑确定性（脏 batch 集是否稳定)
2) 单 batch 复刻: B=1 G=2/8 kv∈{4096,4160,4224,4288,4352,4416} → 是否单独复现
"""
import glob, importlib.machinery, importlib.util, math, os, sys
import torch
import torch_npu

sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
from case_builder import load_cases, parse_case, build_inputs

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c)
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l)
_e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_e)
MLA_SCALE = 0.0721687836487032

def mla_sink(inp, c):
    N1, N2, D, R, BS = c["H"], c["kvh"], c["D"], c["R"], c["BS"]
    B = inp["kvlen_dev"].shape[0]
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)
    return torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=meta, num_query_heads=N1,
        num_key_value_heads=N2, softmax_scale=MLA_SCALE, input_layout="TND",
        sparse_mode=c["sparse_mode"], block_size=BS)

def mla_official(inp, c):
    kw = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
              num_query_heads=c["H"], num_key_value_heads=c["kvh"], input_layout="TND",
              sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
              block_table=inp["bt"], block_size=c["BS"],
              actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
              return_softmax_lse=False)
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)

def mk_case(tag, B, G, kv, seed, H=16, sparse=3, use_mask=True):
    return dict(case_name=tag, scenario="kw9", B=B, G=0, H=H, D=512, R=64, BS=128,
                T=B * G, q_len=[G] * B, kvlen=[kv] * B, pool=sum(math.ceil(x / 128) for x in [kv] * B) + 2,
                bt_w=0, use_mask=use_mask, mask_rows=2048, mask_cols=2048,
                sparse_mode=sparse, gate=0, seed=seed)

def check(tag, cc):
    inp = build_inputs(cc)
    s = mla_sink(inp, cc)[0].float().cpu()
    torch.npu.synchronize()
    o = mla_official(inp, cc).float().cpu()
    torch.npu.synchronize()
    d = (s - o).abs()
    rel = d / o.abs().clamp_min(1e-6)
    print(f"[{tag}] mere={rel.max():.4g} frac>0.02={(rel>0.02).float().mean():.5f}")
    del inp
    torch.npu.empty_cache()
    return s

# 1) C10 三跑
cases = load_cases()
c10 = parse_case(cases["C10_decode_min_g2"])
inp = build_inputs(c10)
runs = []
for i in range(3):
    s = mla_sink(inp, c10)[0]
    torch.npu.synchronize()
    runs.append(s.cpu())
    o = mla_official(inp, c10)[0].float().cpu()
    d = (runs[-1].float() - o).abs()
    rel = d / o.abs().clamp_min(1e-6)
    badb = (rel > 0.02).view(8, 2, 16, -1).float().mean(dim=(1, 2, 3))
    print(f"C10 run{i}: mere={rel.max():.4g} per-batch frac_bad: " + " ".join(f"{x:.4f}" for x in badb.tolist()))
del inp
torch.npu.empty_cache()

# 2) 单 batch 复刻 (C10 形态 G=2)
for kv in [4096, 4160, 4224, 4288, 4352, 4416]:
    check(f"solo_G2_kv{kv}", mk_case(f"solo{kv}", 1, 2, kv, 20260901))

# 3) 单 batch 复刻 (C05 形态 G=8)
for kv in [4096, 4224]:
    check(f"solo_G8_kv{kv}", mk_case(f"solo{kv}", 1, 8, kv, 20260901))
