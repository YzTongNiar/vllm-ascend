"""C01（MLA MTP-verify 形态）门禁——自包含可移植版。
经 PR wrapper 驱动，case_builder/fia_v2_cases.csv 随包。"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PR_TREE = os.environ.get("FIA_GATE_TREE", os.path.join(PKG, "vllm-ascend"))
sys.path.insert(0, HERE)                                  # case_builder/fia_v2_cases.csv/gqa_case_builder
sys.path.insert(0, PR_TREE)
sys.path.insert(0, os.path.join(PR_TREE, "python"))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from case_builder import load_cases, resolve, parse_case, build_inputs  # noqa: E402
from gqa_case_builder import bits_equal  # noqa: E402

# 绕 PEP 660 editable 拦截：直接 ExtensionFileLoader 加载目标树 _C 扩展
# （import vllm_ascend 完整包需要 wheel 安装产物 _build_info，in-place 树没有）
import glob as _glob  # noqa: E402
import importlib.machinery as _machinery  # noqa: E402
import importlib.util as _util  # noqa: E402
_cand = sorted(_glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
assert _cand, f"vllm_ascend_C*.so not found under {PR_TREE}/vllm_ascend"
_ldr = _machinery.ExtensionFileLoader("vllm_ascend_C", _cand[-1])
_spec = _util.spec_from_loader("vllm_ascend_C", _ldr)
_ext = _util.module_from_spec(_spec)
_spec.loader.exec_module(_ext)


def build_fia_v2_sink_metadata(*, num_query_heads, num_key_value_heads,
                               head_dim_qk, head_dim_v, **kw):
    if "actual_seq_qlen" in kw:
        kw["actual_seq_lengths"] = kw.pop("actual_seq_qlen")
    if "actual_seq_kvlen" in kw:
        kw["actual_seq_lengths_kv"] = kw.pop("actual_seq_kvlen")
    return torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        num_query_heads, num_key_value_heads, head_dim_qk, head_dim_v, **kw)


def fused_infer_attention_score_v2_sink(*args, **kw):
    if "metadata" in kw:
        kw["meta_data"] = kw.pop("metadata")
    out, _ = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
        *args, **kw)
    return out

cases = load_cases()
c = parse_case(cases[resolve(cases, "C01")])
inp = build_inputs(c)
BS = c["BS"]
N1, N2 = c["H"], c["kvh"]
D, R = c["D"], c["R"]
scale = 0.0721687836487032

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
print(f"[PR_GATE C01_MLA] out={tuple(on.shape)} finite={finite}"
      f"（bit 对照经外部门禁口径，此处验可运行+有限）")
sys.exit(0 if finite else 1)
