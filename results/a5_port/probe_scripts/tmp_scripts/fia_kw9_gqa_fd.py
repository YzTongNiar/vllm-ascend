"""kw-9: GQA C21-C26 FD 激活状态解码。"""
import glob, importlib.machinery, importlib.util, os, sys
import torch
import torch_npu

sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
from gqa_case_builder import resolve as gqa_resolve, parse_case as gqa_parse, build_inputs as gqa_build

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c)
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l)
_e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_e)

for name in ["C21", "C22", "C23", "C24", "C25", "C26"]:
    c = gqa_parse(gqa_resolve(name))
    inp = gqa_build(c)
    B = inp["kvlen_dev"].shape[0]
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        c["n1"], c["n2"], c["head_dim"], c["head_dim"],
        actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout=c["input_layout"], input_layout_kv="PA",
        rope_head_dim=0, block_size=c["block_size"], aic_core_num=28, aiv_core_num=56)
    torch.npu.synchronize()
    m = meta.cpu().numpy()
    base = m[576:586]
    print(f"GQA_{name}: B={B} n1={c['n1']} n2={c['n2']} FD#={base[3]} uCore={base[4]} uVecFd={base[5]} "
          f"s2spl0={m[6]} gs1spl0={m[8]}")
    del inp, meta
    torch.npu.empty_cache()
