"""kw-9 step1: 逐 case 解码 metadata 的 FD 域字段, 实锤"未收敛 case = FD 激活域"假设。

base[0]=M_BASE base[1]=S_INNER base[2]=M_FD_BASE base[3]=NUM_OF_FD(579)
base[4]=USED_CORE base[5]=USED_VEC_NUM_OF_FD(581) base[6]=S1 base[7]=S2
aic[core][4]=BN2_IDX_OF_FD_HEAD [5]=GS1_IDX_OF_FD [6]=S2_SPLIT_NUM [7]=S2_SPLIT_START
[8]=GS1_SPLIT_NUM [9]=GS1_LAST_PART_SIZE
aiv[core][1]=GS1_IDX_END_OF_FD_HEAD [2]=GS1_IDX_END_OF_FD_HEAD_SPLIT
"""
import glob, importlib.machinery, importlib.util, os, sys
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

# kw-8c 终态 mere (max-rel), None=bit exact; conv = ulp 口径达标
KW8 = {
    "C04": ("1ulp", True), "C05": ("0.14-0.97", False), "C06": ("0.14-0.97", False),
    "C07": ("0.14-0.97", False), "C08": ("1ulp", True), "C09": ("finite-only", None),
    "C10": ("0.14-0.97", False), "C11": ("1ulp", True),
    "C12": ("2.3-4.3", False), "C13": ("2.3-4.3", False), "C14": ("2.3-4.3", False),
    "C15": ("2.3-4.3", False), "C16": ("2.3-4.3", False), "C17": ("2.3-4.3", False),
    "C18": ("1ulp", True), "C19": ("1ulp", True), "C20": ("1ulp", True),
}

cases = load_cases()
names = [n for n in cases if n[:3] in KW8]
names.sort()

print(f"{'case':<20}{'conv':>5} {'mere':>9} | {'M_BASE':>6}{'S_IN':>5}{'M_FD':>5}{'#FD':>4}{'uCore':>6}{'uVecFd':>7}{'S1':>5}{'S2':>6} | fd-heads(bN2:gS1:s2spl:gS1spl)")
mismatch = []
for n in names:
    c = parse_case(cases[n])
    inp = build_inputs(c)
    B = inp["kvlen_dev"].shape[0]
    meta = torch.ops._C_ascend._npu_fused_infer_attention_score_v2_sink_metadata(
        c["H"], c["kvh"], c["D"], c["D"], actual_seq_lengths=inp["qlen_dev"].clone(),
        actual_seq_lengths_kv=inp["kvlen_dev"].clone(), batch_size=B,
        sparse_mode=c["sparse_mode"], input_layout="TND", input_layout_kv="PA",
        rope_head_dim=c["R"], block_size=c["BS"], aic_core_num=28, aiv_core_num=56)
    torch.npu.synchronize()
    m = meta.cpu().numpy()
    base = m[576:586]
    fdheads = []
    maxcore = int(base[4]) if base[4] else 28
    for core in range(min(maxcore, 36)):
        bN2 = m[core * 10 + 4]
        gS1 = m[core * 10 + 5]
        s2spl = m[core * 10 + 6]
        gs1spl = m[core * 10 + 8]
        if bN2 or gS1 or s2spl or gs1spl:
            fdheads.append(f"({bN2}:{gS1}:{s2spl}:{gs1spl})")
    mere_s, conv = KW8[n[:3]]
    fd_on = base[3] > 0
    print(f"{n:<20}{str(conv):>5} {mere_s:>9} | {base[0]:>6}{base[1]:>5}{base[2]:>5}{base[3]:>4}{base[4]:>6}{base[5]:>7}{base[6]:>5}{base[7]:>6} | "
          + (";".join(fdheads[:8]) + ("..." if len(fdheads) > 8 else "")))
    # FD 假设一致性: fd_on == (not conv)
    if conv is not None and fd_on == conv:
        mismatch.append((n, fd_on, conv))
    del inp, meta
    torch.npu.empty_cache()

print("\nFD-hypothesis violations (fd_active == converged):", mismatch if mismatch else "NONE — 假设成立")
