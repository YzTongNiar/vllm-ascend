"""kw-10: C10 残差诊断 — lse 对照 + G=2 vs G=8 判别 + 脏 batch 定位。"""
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

def run_meta(inp, c):
    B = inp["kvlen_dev"].shape[0]
    return torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        c["H"], c["kvh"], c["D"], c["D"], actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=c["R"], block_size=c["BS"], aic_core_num=28, aiv_core_num=56)

def mla_sink(inp, c, lse=False):
    kw = dict(query_rope=inp["qpe"], key_rope=inp["krpool"], atten_mask=inp.get("mask"),
              actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
              block_table=inp["bt"], meta_data=run_meta(inp, c), num_query_heads=c["H"],
              num_key_value_heads=c["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
              sparse_mode=c["sparse_mode"], block_size=c["BS"], return_softmax_lse=lse)
    return torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(inp["q"], inp["kpool"], inp["kpool"], **kw)

def mla_official(inp, c, lse=True):
    kw = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
              num_query_heads=c["H"], num_key_value_heads=c["kvh"], input_layout="TND",
              sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
              block_table=inp["bt"], block_size=c["BS"],
              actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
              return_softmax_lse=lse)
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)

# 1) C10 lse + 脏行分布 ×3
cases = load_cases()
c10 = parse_case(cases["C10_decode_min_g2"])
B, G, H = c10["B"], c10["G"], c10["H"]
inp = build_inputs(c10)
o, o_lse = mla_official(inp, c10)
o = o.float().cpu(); o_lse = o_lse.float().cpu()
torch.npu.synchronize()
s, s_lse = mla_sink(inp, c10, lse=True)
s = s.float().cpu(); s_lse = s_lse.float().cpu()
torch.npu.synchronize()
ratio = (s / o.clamp_min(1e-6)).view(B, G, H, -1)
badrow = (ratio > 1.02).view(B, G, H, -1).any(-1)
print("C10 dirty rows per (batch):", badrow.sum(dim=(1, 2)).tolist())
print("C10 dirty rows per (token):", badrow.sum(dim=(0, 2)).tolist())
dl = (s_lse.view(-1, H) - o_lse.view(-1, H)).abs()
print(f"C10 lse diff max={dl.max().item():.6f}")
for (b, g, h) in badrow.nonzero()[:5].tolist():
    t = b * G + g
    rr = ratio[b, g, h][ (ratio[b,g,h] > 1.02)]
    print(f"  b{b} t{g} h{h}: ratio_med={rr.median():.4f} d_lse={dl[t, h].item():+.6f}")

# 2) G=8 同 kv 对照 (B8 H16 kv 4096-4544 → C10 的 kv 但 G=8)
def mk(tag, G_):
    kv = [4096, 4160, 4224, 4288, 4352, 4416, 4480, 4544]
    return dict(case_name=tag, scenario="kw10", B=8, G=G_, H=16, kvh=1, D=512, R=64, BS=128,
                T=8 * G_, q_len=[G_] * 8, kvlen=kv,
                pool=sum(math.ceil(x / 128) for x in kv) + 2,
                bt_w=max(math.ceil(x / 128) for x in kv), use_mask=True, mask_rows=2048, mask_cols=2048,
                sparse_mode=3, gate=0, seed=20260830)
for G_ in [1, 2, 4, 8]:
    cc = mk(f"g{G_}", G_)
    inp2 = build_inputs(cc)
    o2 = mla_official(inp2, cc, lse=False)[0].float().cpu()
    torch.npu.synchronize()
    meres = []
    for i in range(2):
        s2 = mla_sink(inp2, cc)[0].float().cpu()
        torch.npu.synchronize()
        meres.append(((s2 - o2).abs() / o2.abs().clamp_min(1e-6)).max().item())
    print(f"G={G_}: mere={'/'.join(f'{x:.4g}' for x in meres)}")
    del inp2
    torch.npu.empty_cache()
