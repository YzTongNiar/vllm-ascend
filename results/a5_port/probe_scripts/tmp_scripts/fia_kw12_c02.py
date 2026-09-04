"""kw-12: C02 定性 — 逐 batch(尾宽梯度) 脏分布 + lse + ratio 结构。"""
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

cases = load_cases()
c = parse_case(cases["C02_sig_b_typical"])
B, G, H = c["B"], c["G"], c["H"]
inp = build_inputs(c)
o, o_lse = torch_npu.npu_fused_infer_attention_score_v2(
    inp["q"], inp["kpool"], inp["kpool"],
    query_rope=inp["qpe"], key_rope=inp["krpool"], num_query_heads=H, num_key_value_heads=1,
    input_layout="TND", sparse_mode=0, softmax_scale=MLA_SCALE, block_table=inp["bt"],
    block_size=c["BS"], actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
    return_softmax_lse=True)
o = o.float().cpu(); o_lse = o_lse.float().cpu()
torch.npu.synchronize()
s, s_lse = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"], key_rope=inp["krpool"],
    actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"], block_table=inp["bt"],
    meta_data=run_meta(inp, c), num_query_heads=H, num_key_value_heads=1,
    softmax_scale=MLA_SCALE, input_layout="TND", sparse_mode=0, block_size=c["BS"],
    return_softmax_lse=True)
s = s.float().cpu(); s_lse = s_lse.float().cpu()
torch.npu.synchronize()
rel = ((s - o).abs() / o.abs().clamp_min(1e-6))
print(f"C02: mere={rel.max():.4g} max_abs={(s-o).abs().max():.4g}")
dl = (s_lse.view(-1, H) - o_lse.view(-1, H)).abs()
print(f"lse diff max={dl.max().item():.6f}")
kvs = c["kvlen"]
print(f"{'b':>2}{'kv':>6}{'tail':>6}{'dirty rows':>11}{'mere_b':>9}{'lse_d':>8}")
for b in range(B):
    tail = kvs[b] % 512
    rb = rel.view(B, G, H, -1)[b]
    dirty = (rb > 0.02).view(G, H, -1).any(-1).sum().item()
    print(f"{b:>2}{kvs[b]:>6}{tail:>6}{dirty:>11}{rb.max():>9.4g}{dl.view(B,G,H)[b].max():>8.5f}")
# 脏元素 ratio 方向
r = (s / o.clamp_min(1e-6))
print("ratio<0.98:", (r < 0.98).sum().item(), " ratio>1.02:", (r > 1.02).sum().item())
