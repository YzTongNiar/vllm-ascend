"""kw-12 快照探针读取: 逐循环累计器状态 (C10, batch0 = core0 首任务)。"""
import glob, importlib.machinery, importlib.util, math, os, struct, sys
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

case_name = sys.argv[1] if len(sys.argv) > 1 else "C10_decode_min_g2"
skip_nu = len(sys.argv) > 2 and sys.argv[2] == "skip"
cases = load_cases()
c = parse_case(cases[case_name])
inp = build_inputs(c)
meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
    c["H"], c["kvh"], c["D"], c["D"], actual_seq_lengths=inp["qlen_dev"].clone(),
    actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=c["B"],
    sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
    rope_head_dim=c["R"], block_size=c["BS"], aic_core_num=28, aiv_core_num=56).npu()
meta[604] = 0x600D600D
meta[605] = 0
if skip_nu:
    meta[603] = 0x600D600D
torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"], key_rope=inp["krpool"],
    atten_mask=inp.get("mask"), actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
    block_table=inp["bt"], meta_data=meta, num_query_heads=c["H"],
    num_key_value_heads=c["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
    sparse_mode=c["sparse_mode"], block_size=c["BS"])
torch.npu.synchronize()
m = meta.cpu().numpy()
print(f"== {case_name} skip_nu={skip_nu} (batch0 core0 task, slot=bn2%2)")
for li in range(8):
    base = 700 + li * 24
    loop, bn2, splpos, bidx = int(m[base + 16]), int(m[base + 17]), int(m[base + 18]), int(m[base + 19])
    if loop == 0 and bn2 == 0 and splpos == 0 and bidx == 0 and li > 0:
        continue  # 未写
    vals = []
    for r in range(4):
        row = [struct.unpack("f", struct.pack("I", int(m[base + r * 4 + k])))[0] for k in range(4)]
        vals.append(row)
    any_written = any(v != 0 for row in vals for v in row) or loop != 0 or li == 0
    if not any_written and li > 0:
        continue
    print(f"loop{li}: loop={loop} bn2={bn2} split={splpos} b={bidx}")
    for r in range(4):
        row = vals[r]
        print(f"  r{r}: c0={row[0]:+.5e} c1={row[1]:+.5e} c2={row[2]:+.5e} c3={row[3]:+.5e}")
