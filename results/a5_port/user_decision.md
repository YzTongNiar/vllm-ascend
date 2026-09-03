# user_decision（对 a5_port_feedback.md 的裁决）

- 日期：2026-09-02
- 决策人：user
- 事项：FIA v2 Sink A5 移植 P2 bit 门禁失败后的修复路径（feedback 选项 A/B/C）

## 决定

**选 A 路线（授权 kernel 内修复），执行方式指定：通过 kernel worker agent 改进这个算子。**

## 边界重申（与任务书铁律一致）

1. 契约零变更：torch schema / entry 签名 / 两段式（主算子必带 meta_data）/
   AICPU so 独立命名 `libfia_v2_sink_aicpu_kernels.so` / TND 累计 q-len 契约
2. A3 行为不回退：所有 SoC 差异走条件分支（`ASCEND_COMPUTE_UNIT` / `__CCE_AICORE__` /
   容量常量参数化），A3 路径行为不得改变
3. 验收口径：A5 设备 vs 官方 `torch_npu.npu_fused_infer_attention_score_v2` 逐 bit
   （G1–G5 + C01，门禁脚本同 a5_port_feedback.md 环境备忘）

## 执行要求（指示给 kernel worker agent）

- 方法论：先 FIA 自身差分实证（dump GM workspace，量出 cube 写出范围 vs vector
  读取范围的偏移），再对照 FlashMLA `op_kernel/arch35/`（同仓 A5 已验证同类实现）
  确定 dav-c310 正确写法，最后落条件分支修复
- 已排除假设勿重跑：scale / M 对齐 / workspace 分段（blockDim=coreNum=28 已实锤）/
  编译器 auto-sync / SSBUF 通信模式（均非根因，证据见 MIGRATION_NOTES.md P2 节）
- 每轮重编纪律：`rm gen/fused_infer_attention_score_v2_sink*.done` +
  `rm -rf bin/fused_infer_attention_score_v2_sink` 后 `build.sh --fast`
  （SOC_VERSION=ascend950pr_9599，fatbin 长杆 ~20min/轮）
- 设备：卡 2（`ASCEND_RT_VISIBLE_DEVICES=2`），卡 3 备用

## 追加裁决（2026-09-02）：验收口径 = ulp 口径收尾

- 决策人：user（AskUserQuestion 裁决，选项"接受 ulp 口径收尾（推荐）"）
- 依据：G1/G2 bit 全绿；其余门禁 1-2 ulp 已定性为 SoftmaxFlashV2 与官方
  ProcessVec1Vf 的舍入次序差（非缺陷：单循环 bit 全对、b2 值依赖、lse 差 ≤1 fp32 ulp）；
  bit 补齐最后一环（ProcessVec1Vf dst 布局）静态不可判定，两次探针未钉死
- 生效口径：G1/G2 bit（mere=0.0）；G3-G5/C21-C26 ulp（mere ≤1e-4 且 max_abs ≤2 bf16 ulp）；
  C01/MLA 族 finite+shape；性能 kernel-bound 净赚已证（0.834-0.855×）
- bit 补齐列**遗留项**（非阻塞）：重启路径留档 MIGRATION_NOTES.md P4 kw-7 节

## 追加指示（2026-09-02，同日）：修复完成后的验收流程

kernel worker agent 修复完成后，走**正常精度验收 + 性能测试流程**（已于同日执行完毕，
结果见 GATE_CONSISTENCY.md 全量数据与 perf/perf_a5_result.json）：

1. **精度验收（A5 设备）**：六门禁 + C21–C26 全族 + MLA 抽测（C02/C03/C09）+ q=0 受控快检
2. **性能测试（双方法同机对照）**：sink 两段式 vs 官方，覆盖 X47 成本判据两端
   （S1/S2 kernel-bound + g1_short/kvlen_4096 launch-bound）
3. **交付物更新**：GATE_CONSISTENCY.md 双列终表、cannbot_handoff.json（verification/perf 节）、
   迁移笔记终态、分支提交

## 指针

- 证据链：`results/a5_port/MIGRATION_NOTES.md`（P0–P4 全程）+ `a5_port_feedback.md`
- 分支：`feat/fia-v2-sink-a5` @ `22faaa0d1`（功能态 = kw-4 三刀修复；vllm-ascend clone，本包内）
- handoff：`vllm-ascend/handoff/cannbot_handoff.json` a5_port 节
