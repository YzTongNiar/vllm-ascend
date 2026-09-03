import glob, importlib.machinery, importlib.util, math, os, sys, torch, torch_npu
sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
from gqa_case_builder import parse_case, resolve, build_inputs
PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c[-1])
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l); _e = importlib.util.module_from_spec(_s); _s.loader.exec_module(_e)
c = parse_case(resolve("C21"))
inp = build_inputs(c)
# q=0 覆盖
qz = torch.zeros_like(inp["q"])
ob, _ = torch_npu.npu_fused_infer_attention_score_v2(
    qz, inp["kpool"], inp["vpool"], num_query_heads=c["n1"], num_key_value_heads=c["n2"],
    input_layout="TND", sparse_mode=0, softmax_scale=inp["scale"], block_table=inp["bt"],
    block_size=c["block_size"], actual_seq_qlen=inp["qlen_host"], actual_seq_kvlen=inp["kvlen_host"],
    return_softmax_lse=False)
meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
    c["n1"], c["n2"], c["head_dim"], c["head_dim"],
    actual_seq_lengths=inp["qlen_dev"], actual_seq_lengths_kv=inp["kvlen_dev"],
    input_layout="TND", input_layout_kv="PA", sparse_mode=0, block_size=c["block_size"], rope_head_dim=0)
on_, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    qz, inp["kpool"], inp["vpool"], actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
    block_table=inp["bt"], meta_data=meta, num_query_heads=c["n1"], num_key_value_heads=c["n2"],
    softmax_scale=inp["scale"], input_layout="TND", sparse_mode=0, block_size=c["block_size"],
    return_softmax_lse=False)
torch.npu.synchronize()
d = (on_.float() - ob.float()).abs()  # [T,N1,D]
print("C21 q=0:", c["q_len_per_req"], c["kv_len_per_req"], "gS=", c["gsize"])
# 按请求/头聚合
t = 0
for i, ql in enumerate(c["q_len_per_req"]):
    row = []
    for h in range(c["n1"]):
        row.append(f"h{h}:{d[t:t+ql, h].max().item():.3g}")
    print(f" b{i}(kv={c['kv_len_per_req'][i]}): " + " ".join(row))
    t += ql
print(f"total max={d.max().item():.4g}")
