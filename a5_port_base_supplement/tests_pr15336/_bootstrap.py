"""自包含引导：不依赖 vllm_ascend 包解析（PEP 660 editable 会拦截 import），
直接 ExtensionFileLoader 加载指定树内的 vllm_ascend_C*.so，
使 torch.ops._C_ascend.* 指向 FIA_GATE_TREE（默认随包 vllm-ascend/）的构建。"""
import glob
import importlib.machinery
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PR_TREE = os.environ.get("FIA_GATE_TREE", os.path.join(PKG, "vllm-ascend"))

sys.path.insert(0, HERE)      # gqa_case_builder / case_builder / fia_v2_cases.csv


def load_ext():
    cands = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))
    if not cands:
        raise RuntimeError(
            f"vllm_ascend_C*.so not found under {PR_TREE}/vllm_ascend — "
            "构建未完成或 FIA_GATE_TREE 指错树")
    path = cands[-1]
    loader = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", path)
    spec = importlib.util.spec_from_loader("vllm_ascend_C", loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
