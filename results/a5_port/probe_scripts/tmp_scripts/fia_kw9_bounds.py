"""kw-9: 逐核 s2 任务边界 dump, 关联 dirty。"""
import glob, importlib.machinery, importlib.util, math, os, sys
import torch
import torch_npu

sys.path.insert(0, "/home/t00886357/a5_port_base_supplement/tests_pr15336")
from case_builder import build_inputs

PR_TREE = os.environ.get("FIA_GATE_TREE", "/home/t00886357/fia_sink_a5_port_base/vllm-ascend")
_c = sorted(glob.glob(os.path.join(PR_TREE, "vllm_ascend", "vllm_ascend_C*.so")))[-1]
_l = importlib.machinery.ExtensionFileLoader("vllm_ascend_C", _c)
_s = importlib.util.spec_from_loader("vllm_ascend_C", _l)
_e = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_e)

def mk(tag, kvs):
    B = len(kvs)
    return dict(case_name=tag, scenario="kw9", B=B, G=8, H=16, kvh=1, D=512, R=64, BS=128,
                T=B * 8, q_len=[8] * B, kvlen=list(kvs),
                pool=sum(math.ceil(x / 128) for x in kvs) + 2,
                bt_w=max(math.ceil(x / 128) for x in kvs), use_mask=True, mask_rows=2048, mask_cols=2048,
                sparse_mode=3, gate=0, seed=20260901)

def dump(tag, cc):
    inp = build_inputs(cc)
    B = cc["B"]
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        cc["H"], cc["kvh"], cc["D"], cc["D"], actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=cc["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=cc["R"], block_size=cc["BS"], aic_core_num=28, aiv_core_num=56)
    torch.npu.synchronize()
    m = meta.cpu().numpy()
    u = m[580]
    print(f"== {tag} uCore={u}")
    prev_s2 = 0
    prev_bn2 = 0
    for core in range(u):
        bn2e, gs1e, s2e = m[core*10+1], m[core*10+2], m[core*10+3]
        if bn2e != prev_bn2:
            prev_s2 = 0  # 新 batch
        rng = s2e - prev_s2
        loops = rng / 512.0
        tail = rng % 512
        print(f"  core{core:02d}: bN2End={bn2e} s2=[{prev_s2},{s2e}) len={rng} loops={loops:.3f} tail={tail}")
        prev_s2, prev_bn2 = s2e, bn2e
    del inp
    torch.npu.empty_cache()

dump("B8_onlyB1_8256 (DIRTY)", mk("a", [8192, 8256] + [8192] * 6))
dump("B8_onlyB1_8208 (CLEAN)", mk("b", [8192, 8208] + [8192] * 6))
