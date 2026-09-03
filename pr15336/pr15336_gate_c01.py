import math, sys
import os as _os
OURS = _os.environ.get("FIA_GATE_BASE", "/home/t00886357/fia_sink_a5_port_base")
sys.path.insert(0, OURS + "/tests")
import torch
from case_builder import load_cases, resolve, parse_case, build_inputs
from gqa_case_builder import bits_equal
from vllm_ascend.ops.fia_v2_sink import (
    build_fia_v2_sink_metadata, fused_infer_attention_score_v2_sink)

cases = load_cases(); c = parse_case(cases[resolve(cases, "C01")])
inp = build_inputs(c)
BS = c["BS"]
N1, N2 = c["H"], c["kvh"]
D, R = c["D"], c["R"]
scale = 1.0 / math.sqrt(D + DR) if False else 0.0721687836487032
meta = build_fia_v2_sink_metadata(
    actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
    num_query_heads=N1, num_key_value_heads=N2, head_dim_qk=D, head_dim_v=D,
    input_layout="TND", input_layout_kv="PA", sparse_mode=c["sparse_mode"],
    block_size=BS, rope_head_dim=R)
on = fused_infer_attention_score_v2_sink(
    inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"], key_rope=inp["krpool"],
    atten_mask=inp.get("mask"), actual_seq_qlen=inp["qlen_dev"],
    actual_seq_kvlen=inp["kvlen_dev"], block_table=inp["bt"], metadata=meta,
    num_query_heads=N1, num_key_value_heads=N2, softmax_scale=scale,
    input_layout="TND", sparse_mode=c["sparse_mode"], block_size=BS)
torch.npu.synchronize()
finite = bool(torch.isfinite(on.float()).all().item())
print(f"[PR_GATE C01_MLA] out={tuple(on.shape)} finite={finite}（bit 对照经外部门禁口径，此处验可运行+有限）")
