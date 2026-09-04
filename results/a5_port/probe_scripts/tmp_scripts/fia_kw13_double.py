"""kw-13 双读判别读取: C10 batch0 — 首读 raw [820+] vs 晚读 raw [800+]。"""
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

cases = load_cases()
c10 = parse_case(cases["C10_decode_min_g2"])
inp = build_inputs(c10)
meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
    c10["H"], c10["kvh"], c10["D"], c10["D"], actual_seq_lengths=inp["qlen_dev"].clone(),
    actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=c10["B"],
    sparse_mode=c10["sparse_mode"], input_layout="TND", input_layout_kv="PA",
    rope_head_dim=c10["R"], block_size=c10["BS"], aic_core_num=28, aiv_core_num=56).npu()
meta[606] = 0x600D600D
meta[602] = 0
torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"], key_rope=inp["krpool"],
    atten_mask=inp["mask"], actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
    block_table=inp["bt"], meta_data=meta, num_query_heads=c10["H"],
    num_key_value_heads=c10["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
    sparse_mode=c10["sparse_mode"], block_size=c10["BS"])
torch.npu.synchronize()
m = meta.cpu().numpy()

def rd(base):
    return [[struct.unpack("f", struct.pack("I", int(m[base + r * 4 + k])))[0] for k in range(4)] for r in range(2)]

for cp in range(4):
    for parity in range(2):
        b_early = 820 + parity * 40 + 0  # 首读采样区按 (blockIdx%2) 区分
        b_late = 800 + parity * 40
        # 实际区域: early=[820+parity*40], late=[800+parity*40]; corePair 由 dump 的 loop/bn2 溯源
    # 简化: 直接打印 4 个区
for tag, base in [("late-even", 800), ("late-odd", 840), ("early-even", 820), ("early-odd", 860)]:
    vals = rd(base)
    prov = (int(m[base + 8]), int(m[base + 9]) if base + 9 < 1024 else -1)
    nz = any(v != 0 for row in vals for v in row)
    print(f"{tag}: loop={prov[0]} bn2={prov[1]} {'DATA' if nz else '(empty)'}")
    for r in range(2):
        print(f"  r{r}: {['%+.6e' % v for v in vals[r]]}")
# 一致性: early vs late 同 parity
for parity, tag in [(0, "even"), (1, "odd")]:
    e = rd(820 + parity * 40)
    l = rd(800 + parity * 40)
    same = all(abs(e[r][k] - l[r][k]) < 1e-9 * max(1, abs(e[r][k])) for r in range(2) for k in range(4))
    print(f"{tag}: early==late (bit-tolerance): {same}")
    for r in range(2):
        for k in range(4):
            if e[r][k] != l[r][k]:
                print(f"  DIFF r{r} c{k}: early={e[r][k]:+.6e} late={l[r][k]:+.6e} ratio={l[r][k]/e[r][k] if e[r][k] else float('nan'):.6f}")
