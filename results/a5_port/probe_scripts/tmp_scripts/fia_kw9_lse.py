"""kw-9: C07 lse 对照 — 分母错(lse同比例偏) vs 分子错(lse对/out偏)。跨进程稳定性。"""
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
    N1, N2, D, R, BS = c["H"], c["kvh"], c["D"], c["R"], c["BS"]
    B = inp["kvlen_dev"].shape[0]
    return torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)

cases = load_cases()
name = sys.argv[1] if len(sys.argv) > 1 else "C07_ctx8k"
c = parse_case(cases[name])
B, G, H = c["B"], c["G"], c["H"]
inp = build_inputs(c)

kw_o = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
            num_query_heads=c["H"], num_key_value_heads=c["kvh"], input_layout="TND",
            sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
            block_table=inp["bt"], block_size=c["BS"],
            actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
            return_softmax_lse=True)
if c["use_mask"]:
    kw_o["atten_mask"] = inp["mask"]
o, o_lse = torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw_o)
o = o.float().cpu(); o_lse = o_lse.float().cpu()
torch.npu.synchronize()

kw_s = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
            actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
            block_table=inp["bt"], meta_data=run_meta(inp, c), num_query_heads=c["H"],
            num_key_value_heads=c["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
            sparse_mode=c["sparse_mode"], block_size=c["BS"], return_softmax_lse=True)
if c["use_mask"]:
    kw_s["atten_mask"] = inp["mask"]
s, s_lse = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    inp["q"], inp["kpool"], inp["kpool"], **kw_s)
torch.npu.synchronize()
s = s.float().cpu()
s_lse = s_lse.float().cpu() if s_lse is not None and s_lse.numel() else None

print(f"{name}: sink lse shape={tuple(s_lse.shape) if s_lse is not None else None} off lse shape={tuple(o_lse.shape)}")
ratio = (s / o.clamp_min(1e-6)).view(B, G, H, -1)
badrow = (ratio > 1.02).view(B, G, H, -1).any(-1)
print("dirty rows:", badrow.sum().item(), "batches:", sorted(set(badrow.nonzero()[:, 0].tolist())))

# lse 布局: official [T?, H, 1] or [B,S,H]? 打印并按 TND [T,H] 展开
o_l = o_lse.view(-1, c["H"]) if o_lse.numel() == (B * G) * c["H"] else o_lse.reshape(-1, c["H"])
s_l = s_lse.view(-1, c["H"]) if s_lse.numel() == (B * G) * c["H"] else s_lse.reshape(-1, c["H"])
dl = (s_l - o_l)
print(f"lse diff: max={dl.abs().max().item():.6f} mean={dl.abs().mean().item():.6g}")
# 脏行的 lse vs 干净行的 lse
for (b, g, h) in badrow.nonzero()[:6].tolist():
    t = b * G + g
    print(f"  b{b} t{g} h{h}: out_ratio={ratio[b, g, h].median():.4f} "
          f"lse_off={o_l[t, h].item():.4f} lse_sink={s_l[t, h].item():.4f} d_lse={dl[t, h].item():+.6f}")
# 干净行对照
clean = ~badrow
for (b, g, h) in clean.nonzero()[:3].tolist():
    t = b * G + g
    print(f"  clean b{b} t{g} h{h}: lse_off={o_l[t, h].item():.4f} lse_sink={s_l[t, h].item():.4f} d={dl[t, h].item():+.6f}")
