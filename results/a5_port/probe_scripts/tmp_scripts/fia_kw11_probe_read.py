"""kw-11 探针读取: metadata[601]=magic, [602]=batch → 跑 C10 → 解码 [900..995]。"""
import glob, importlib.machinery, importlib.util, os, struct, sys
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
MAGIC = 0x600D600D

cases = load_cases()
sel_batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
c10 = parse_case(cases["C10_decode_min_g2"])
inp = build_inputs(c10)
B = c10["B"]

meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
    c10["H"], c10["kvh"], c10["D"], c10["D"], actual_seq_lengths=inp["qlen_dev"].clone(),
    actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
    sparse_mode=c10["sparse_mode"], input_layout="TND", input_layout_kv="PA",
    rope_head_dim=c10["R"], block_size=c10["BS"], aic_core_num=28, aiv_core_num=56)
# 激活探针: [601]=magic, [602]=batch
meta_dev = meta.npu()
meta_dev[601] = MAGIC
meta_dev[602] = sel_batch
s = torch.ops._C_ascend.npu_fused_infer_attention_score_v2_sink(
    inp["q"], inp["kpool"], inp["kpool"], query_rope=inp["qpe"], key_rope=inp["krpool"],
    atten_mask=inp["mask"], actual_seq_qlen=inp["qlen_dev"], actual_seq_kvlen=inp["kvlen_dev"],
    block_table=inp["bt"], meta_data=meta_dev, num_query_heads=c10["H"],
    num_key_value_heads=c10["kvh"], softmax_scale=MLA_SCALE, input_layout="TND",
    sparse_mode=c10["sparse_mode"], block_size=c10["BS"])[0]
torch.npu.synchronize()
m = meta_dev.cpu().numpy()
print(f"probe batch={sel_batch}")
for core in range(4):
    base = 900 + core * 24
    vals = [struct.unpack("f", struct.pack("I", int(m[base + r * 4 + k])))[0] for r in range(4) for k in range(4)]
    divs = [struct.unpack("f", struct.pack("I", int(m[base + 16 + r])))[0] for r in range(4)]
    loop, bn2, splpos, drc = int(m[base + 20]), int(m[base + 21]), int(m[base + 22]), int(m[base + 23])
    print(f"corePair{core}: loop={loop} bn2InCore={bn2} splitPos={splpos} dealM={drc}")
    for r in range(4):
        row = vals[r * 4:r * 4 + 4]
        if row[0] != 0 and all(abs(x) < 1e30 for x in row):
            print(f"  row{r}: c0={row[0]:.4f} c1={row[1]:.4f} c2={row[2]:.4f} c3={row[3]:.4f} "
                  f"| c2/c0={row[2]/row[0] if row[0] else float('nan'):.4f} c3/c0={row[3]/row[0] if row[0] else float('nan'):.4f} div={divs[r]:.4f}")
        else:
            print(f"  row{r}: <zero/invalid> {row}")
