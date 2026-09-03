"""kw-9c 构建后终验: MLA 16 case 快扫 (每 case 2 跑)。"""
import glob, importlib.machinery, importlib.util, math, os, sys
import torch
import torch_npu

sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
from case_builder import load_cases, parse_case, build_inputs

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
print("ext:", _c)
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

def mla_sink(inp, c):
    return torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"],
        key_rope=inp["krpool"], atten_mask=inp.get("mask"),
        actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
        block_table=inp["bt"], meta_data=run_meta(inp, c), num_query_heads=c["H"],
        num_key_value_heads=c["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
        sparse_mode=c["sparse_mode"], block_size=c["BS"])

def mla_official(inp, c):
    kw = dict(query_rope=inp["qpe"], key_rope=inp["krpool"],
              num_query_heads=c["H"], num_key_value_heads=c["kvh"], input_layout="TND",
              sparse_mode=c["sparse_mode"], softmax_scale=MLA_SCALE,
              block_table=inp["bt"], block_size=c["BS"],
              actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
              return_softmax_lse=False)
    if c["use_mask"]:
        kw["atten_mask"] = inp["mask"]
    return torch_npu.npu_fused_infer_attention_score_v2(inp["q"], inp["kpool"], inp["kpool"], **kw)[0]

cases = load_cases()
names = [n for n in cases if n[:3] in {f"C{i:02d}" for i in list(range(4, 9)) + list(range(10, 21))}]
names.sort()
print(f"{'case':<22}{'mere':>22}  {'bit_eq':>7}")
fails = []
for n in names:
    c = parse_case(cases[n])
    inp = build_inputs(c)
    o = mla_official(inp, c).float().cpu()
    torch.npu.synchronize()
    meres, outs = [], []
    for i in range(2):
        s = mla_sink(inp, c)[0]
        torch.npu.synchronize()
        outs.append(s.cpu())
        meres.append(((s.float().cpu() - o).abs() / o.abs().clamp_min(1e-6)).max().item())
    det = torch.equal(outs[0], outs[1])
    m = "/".join(f"{x:.4g}" for x in meres)
    ok = meres[-1] <= 0.0079
    if not ok:
        fails.append(n)
    print(f"{n:<22}{m:>22}  {str(det):>7}  {'OK' if ok else 'FAIL'}")
    del inp
    torch.npu.empty_cache()
print("\nFails:", fails if fails else "NONE — 全族 1ulp")
