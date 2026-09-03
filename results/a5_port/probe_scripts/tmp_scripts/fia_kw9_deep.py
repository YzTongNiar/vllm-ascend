"""kw-9: Bug B 深挖 — 脏行代数结构。假设检验: out_sink/out_off ≈ 1/(1-w), w≈1/s2SplitNum?
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

def run_meta(inp, c):
    N1, N2, D, R, BS = c["H"], c["kvh"], c["D"], c["R"], c["BS"]
    B = inp["kvlen_dev"].shape[0]
    return torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        N1, N2, D, D, actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=R, block_size=BS, aic_core_num=28, aiv_core_num=56)

def mla_sink(inp, c):
    meta = run_meta(inp, c)
    return torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=meta, num_query_heads=c["H"],
        num_key_value_heads=c["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
        sparse_mode=c["sparse_mode"], block_size=c["BS"])

def mla_official(inp, c):
    kw = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
              num_query_heads=c["H"], num_key_value_heads=c["kvh"], input_layout="TND",
              sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
              block_table=inp["bt"], block_size=c["BS"],
              actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
              return_softmax_lse=True)
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)

cases = load_cases()
name = sys.argv[1] if len(sys.argv) > 1 else "C07_ctx8k"
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
c = parse_case(cases[name])
B, G, H = c["B"], c["G"], c["H"]
inp = build_inputs(c)
meta = run_meta(inp, c).cpu().numpy()
# 每 batch 的 s2SplitNum: 从 aic 表 bN2 递增段推
bN2 = [meta[i * 10 + 4] for i in range(28)]
spl = [meta[i * 10 + 6] for i in range(28)]
print(f"{name} B={B} G={G} H={H} aic FD表(bN2,s2spl): " + str(list(zip(bN2, spl))[:28]))

o, lse = mla_official(inp, c)
o = o.float().cpu()
torch.npu.synchronize()
prev_bad = None
for r in range(runs):
    s = mla_sink(inp, c)[0].float().cpu()
    torch.npu.synchronize()
    ratio = (s / o.clamp_min(1e-6)).view(B, G, H, -1)
    bad = (ratio > 1.02)  # 元素级
    badrow = bad.view(B, G, H, -1).any(-1)  # [B,G,H]
    print(f"run{r}: bad rows={badrow.sum().item()}/{B*G*H}")
    if prev_bad is not None:
        print(f"  vs prev: same={(badrow == prev_bad).sum().item()} only_now={(badrow & ~prev_bad).sum().item()} only_prev={(prev_bad & ~badrow).sum().item()}")
    prev_bad = badrow
    # 每个 dirty 行的 ratio 中位数 → 隐含缺失质量 w = 1 - 1/ratio
    if badrow.any():
        rows = badrow.nonzero()
        ws = []
        for b, g, h in rows[:12].tolist():
            rr = ratio[b, g, h][bad.view(B, G, H, -1)[b, g, h]]
            ws.append((b, g, h, rr.median().item()))
        for b, g, h, med in ws:
            print(f"  b{b} t{g} h{h}: ratio_med={med:.4f} implied_missing_w={1 - 1 / med:.4f}")
del inp
