# a5_port 阻塞决策登记（按任务书 §7，等确认）

日期：2026-09-02　分支：`feat/fia-v2-sink-a5` @ `d5c7216ca`

## 事项：P2 验收未达 bit 口径，需决策后续路径

**已完成**（P0/P1 全绿）：
- 工具链确认（9.1.0 支持 ascend950）、A5 规格勘察、arch35 归位
- A5 全量编译零错误（SOC_VERSION=ascend950pr_9599），OPP/AICPU so/torch 扩展产物齐备
- 双算子在 A5 设备端到端跑通；metadata AICPU 任务表计算正确；C01 MLA 可运行性门禁通过

**未完成**（P2 bit 门禁 G1–G5 确定性失败，mere 0.27–1.81）：

根因已定位但未修复：主算子 kernel cube→vector 的 P 矩阵传递路径
（L0C fixpipe → GM workspace → vector softmax 读取）在 dav-c310 上行为不等价。
证据链（详见 MIGRATION_NOTES.md P2 节）：
- q=0 受控实验：官方 out=V 均值（数学正确），本实现 softmax 权重非均匀（P 污染）
- 错误确定性 + 按行结构性分布（尾部正确区 = 写/读重叠特征）
- 已排除：scale/M 对齐/workspace 分段错位/metadata 核数/编译器同步（auto-sync）/SSBUF 通信模式
  （SSBUF 与 auto-sync 仅使 mere 1.0→0.27，非修复）

**为什么停下**：修复需进入 kernel 内部（fixpipe 参数/写出偏移/事件链的 dav-c310 适配），
超出本任务"不改算子行为语义、kernel 缺陷只定位"的边界；且逐项试错的编译-验证循环
成本高（每轮 ~20min fatbin 重编）。

## 建议选项（请决策）

| 选项 | 内容 | 成本估计 |
|---|---|---|
| A | 授权 kernel 内修复：对照 FlashMLA arch35 的 P 传递实现逐段核对 FIA kernel（FixpipeOut.h / buffer_mix_core.h / vector 读 P 偏移），dump GM workspace 确认写出范围后修 | 0.5–2 天，每轮重编 20min |
| B | 按当前状态交付：A5 编译链路 + 链路可运行 + C01 口径 + 根因分析，kernel 修复立独立任务 | 0（本文件即交付） |
| C | 混合：先交付 A，同时开 kernel debug 子任务（我方继续定位，产出 patch 草案供评审） | 与 A 并行 |

## 另两项独立登记

1. **910_93 回归未实测**：本机 8 卡全 950PR，无 910_93。A3 行为不回退以静态论证替代
   （唯一非条件改动 = 数组容量 26→36，循环上界来自 metadata usedCoreNum≤A3 核数 24，
   行为零变化；其余改动均在 ascend950 条件分支）。建议回 A3 环境复跑全量门禁终验。
2. **补充包脚本 4 处适配**（非算子逻辑）：metadata/主算子直调参数名归一（lengths/meta_data，
   2 个包装函数内单点转换）、C01 重复传参 bug、C01 改 ExtensionFileLoader 直载——
   建议反馈补充包维护方合入。

## 环境备忘（复现）

```bash
cd /home/t00886357/a5_port_base_supplement/tests_pr15336
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
export FIA_GATE_TREE=/home/t00886357/fia_sink_a5_port_base/vllm-ascend
OPP=$FIA_GATE_TREE/vllm_ascend/_cann_ops_custom
export ASCEND_CUSTOM_OPP_PATH=$OPP:$OPP/vendors/custom_transformer
python3 pr15336_gate.py mqa_c27        # 等
# 重编（改 kernel/选项后）：
#   rm -f <tree>/csrc/build/binary/ascend950/gen/fused_infer_attention_score_v2_sink*.done
#   rm -rf <tree>/csrc/build/binary/ascend950/bin/fused_infer_attention_score_v2_sink
#   cd <base> && SOC_VERSION=ascend950pr_9599 MAX_JOBS=64 bash build.sh --fast
```
