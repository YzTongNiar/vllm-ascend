"""kw-9: 尾块尺寸扫描 — B8 pos1 kv=8192+tail, tail ∈ 16..496。"""
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

def mk(tag, kvs, seed=20260901):
    B = len(kvs)
    return dict(case_name=tag, scenario="kw9", B=B, G=8, H=16, kvh=1, D=512, R=64, BS=128,
                T=B * 8, q_len=[8] * B, kvlen=list(kvs),
                pool=sum(math.ceil(x / 128) for x in kvs) + 2,
                bt_w=max(math.ceil(x / 128) for x in kvs), use_mask=True, mask_rows=2048, mask_cols=2048,
                sparse_mode=3, gate=0, seed=seed)

for tail in [16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448, 496]:
    cc = mk(f"t{tail}", [8192, 8192 + tail] + [8192] * 6)
    inp = build_inputs(cc)
    o = mla_official(inp, cc).float().cpu()
    torch.npu.synchronize()
    meres = []
    for i in range(2):
        s = mla_sink(inp, cc)[0].float().cpu()
        torch.npu.synchronize()
        meres.append(((s - o).abs() / o.abs().clamp_min(1e-6)).max().item())
    print(f"tail={tail:3d} kv={8192+tail}: mere={'/'.join(f'{x:.4g}' for x in meres)}")
    del inp
    torch.npu.empty_cache()
