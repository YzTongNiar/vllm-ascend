# FIA v2 Sink 双算子 A3→A5 移植交接文档

> 交接日期：2026-08-31　状态：待启动（P0 勘察起）
> 任务：将 FIA v2 Sink 双算子（主算子 AICore + metadata AICPU）从 A3（Ascend910_93）
> 移植到 A5（Ascend950），保持功能与精度等价，契约零变更。
> 本文档为唯一 briefing；执行 session 按此开工，不再依赖本 session 上下文。

---

## 0. 权威基线（先读）

| 项 | 内容 |
|---|---|
| 权威源 | `/home/t00886357/npu_fused_infer_attention_score_v2_sink/vllm-ascend/`，分支 `feat/fia-v2-sink` @ `ded354d3d` |
| 算子目录 | `csrc/attention/fused_infer_attention_score_v2_sink/`（主算子，扁平布局：op_host/op_kernel/op_api/torch_binding）、`…_metadata/`（AICPU 前置：op_kernel_aicpu/op_graph/op_api/op_host/config.ini）、`…_common/`（`upstream_common/` + `attention_common/` 公共库） |
| 构建入口 | 工作区根 `build.sh`（清 `csrc/build` 全量 / `--fast` 增量；env：`source set_env.sh; SOC_VERSION=ascend910_9382 MAX_JOBS=64 COMPILE_CUSTOM_KERNELS=1`） |
| 门禁 | `tests/pr15336/pr15336_gate{,_c01}.py`、`pr15336_graph_probe.py`（FIA_GATE_TREE 可配）+ `tests/gqa_case_builder.py`、`check_gqa.py`、`case_builder.py`（C01–C26） |
| 行为基准 | 官方 `torch_npu.npu_fused_infer_attention_score_v2`（同机 bit 对照；MERE_golden 为 P2 口径不作门禁） |
| 契约红线（零变更） | torch schema / entry 签名 / 两段式（主算子必带 meta_data）/ AICPU so 独立命名 `libfia_v2_sink_aicpu_kernels.so` / TND **累计** q-len 契约 |

**⚠️ `op_src/` 为存档只读**（omni 谱系 provenance 留档）——A3 时代的独立构建链，与本任务无关，勿在其上改动。

## 1. 背景与已知事实（已勘察，节省你半天）

1. **构建系统已支持 A5**：`csrc/CMakeLists.txt:37-39` 的 `ASCEND_ALL_COMPUTE_UNIT` /
   `SOC_VERSION_LIST` 已含 **`ascend950`**；`setup.py:85-118` 有 Ascend950 chip-name 处理。
   → 构建目标层面的 A5 标识即 **`ascend950`**。
2. **仓内尚无算子做过 A5 适配**（attention/CMakeLists 无 ascend950 条件编译先例）——
   本任务将是首个，无现成算子模板可抄；但 AscendC 构建框架（ascendc_kernel_cmake）
   对多 SoC 的支持是通用的。
3. **本算子内核为扁平布局（无 arch 子目录）**，当前按 910_93 单 SoC 编译；仓内其他算子
   的 arch 约定是 `op_kernel/archNN/`（arch35=910_93，arch22/32 见 sparse_flash_attention）。
   A5 的 arch 代号需 P0 确认。
4. **本机工具链为 CANN 9.1.0**（/usr/local/Ascend/cann-9.1.0）。ascend950 编译目标是否被
   9.1.0 的 opc/ascendc_kernel_cmake 支持 = **P0 首个确认项**；若不支持，需先申请/安装
   支持 A5 的 CANN 版本（此项阻塞后续一切，优先做）。
5. **AICPU 侧**：`aicpu-config.cmake`/`aicpu_modules` 在 ascendc_kernel_cmake 中存在；
   AICPU so 聚合 + RENAME 契约名机制已在本仓验证（见 metadata CMakeLists 注释）。

## 2. A3 特有假设清单（移植排查重点，逐项核对）

以下为本算子源码中已识别的、可能隐含 A3（910_93）硬件假设的位置：

| # | 位置 | 内容 | A5 风险 |
|---|---|---|---|
| H1 | `…_metadata/op_kernel_aicpu/*.h`：`AIC_CORE_NUM=36`、`AIV_CORE_NUM=72` | metadata 三段表的**合同维度上限** | A5 核数若超出（更多 AIC/AIV），表溢出；若不足，浪费但安全。核对 A5 规格后调整或改动态 |
| H2 | `csrc/attention/…/op_kernel/fia_block_cube_nonquant_gqa_sink.h` 的 **L1 分区常量含 D 依赖表达式**（实测）：`L1_Q_SIZE = S2_BASICSIZE_IS_1024 ? 128*192 : 256*192`、`L1_V_SIZE=256*128`、`L1_KP_SIZE=128*256`、`L1_SINK_KP_SIZE=128*192`；且 L0/L1 各 half 精确等于特定 D×分块组合的 exact-fit（如 L1_KP=128×256=64KB 注释 "K: 128*192 or 128*256"） | A5 的 L1/L0 容量与核内分块公式绑定这些 D 相关 exact-fit 值；**round_5 的 L0B 修复（kCap=min(128,(L0B_PP_SIZE/headDim)&~15)）已参数化，但 L1 各分区尺寸是硬编码** | A5 规格核对后按新容量重推；D 依赖表达式需矩阵化排查（grep `128 * 192\|256 * 128\|128 * 256`） |
| H3 | `fia_kernel_nonquant_sink.cpp:161` 等：`__CCE_AICORE__ == 200/310` 版本宏 | 910_93 的 AICORE 版本号 ≠ A5 的；相关条件分支需按 A5 版本号补臂或确认走默认 | 中 |
| H4 | tikcfw include 路径含 `dav_c220`（910_93 核架构代号） | 框架头随工具链走，一般无需改 | 低 |
| H5 | `op_kernel/` 无 arch 子目录 | A5 若要求 arch 目录约定（archNN），需按 ops-build 约定补 | 中（见 sparse_flash_attention 的 arch22/arch35 双目录先例） |
| H6 | `fia_kernel_*.h`/`fia_block_*.h` 内 `AscendC::` 指令组合（load2d/load3d/mmad/fixpipe） | A5 指令集/时序若变化，需重验 | 编译通过≠正确，必须过门禁 |

**排查方法**：grep `__CCE_AICORE__|dav_c|arch35|910_93|9391` 于算子树；对照 A5 的
工具链宏与规格文档逐项裁决。

## 3. 分阶段计划

### P0 环境与规格勘察（半天，阻塞项优先）
- [ ] 确认 A5 的 SoC 标识串（`ascend950`？）与所需 CANN 版本；确认 9.1.0 工具链可否编
      ascend950 目标（`grep -rn ascend950 /usr/local/Ascend/cann-9.1.0/tools/tikcpp/`；
      或查 `soc_list`）——**不可用则先停，申请 A5 CANN 工具链**
- [ ] 拿到 A5 硬件规格：AIC/AIV 核数、UB 容量、L0A/L0B/L0C/L1 容量（决定 H1/H2）
- [ ] 确认验证手段：有无 A5 设备卡 / 近似仿真环境（决定 P2 的验收口径）

### P1 编译打通（1–2 天）
- [ ] `build.sh` 增加 A5 档：`SOC_VERSION=ascend910_9382` → 增加 `ascend950` 分支
- [ ] 逐个解决 H1–H5 编译错误/条件分支（每修一处记录到迁移笔记）
- [ ] 验收：A5 目标全量编译零错误，产出 OPP 布局（kernel/ascend950/<op>/ + config json）

### P2 功能等价验证（2–3 天）
- [ ] 门禁迁移：六门禁（mqa_c27/c28、C21、kvlen 4096/32768、b2_248）+ C01 MLA，
      `FIA_GATE_TREE` 指向 A5 构建产物
- [ ] 验收口径：有 A5 设备 → vs 官方 bit 对照；仅有仿真 → MERE 口径（如 §8 P2 门限
      2⁻⁷）并在 handoff 显式登记“非 bit 口径”
- [ ] 若 A5 无设备且无仿真：以 910_93 回归 + 静态审查交付，并在 handoff 标记
      “A5 行为未实测”

### P3 收尾（1 天）
- [ ] 全量回归（C01–C26 于 910_93 不回退——**A5 分支不得影响 A3 行为**，条件编译隔离）
- [ ] 交接：包路径、SHA256、迁移笔记（每处架构假设的裁决记录）、遗留清单

## 4. 铁律（不变）

- 契约零变更：torch schema / entry 签名 / 两段式 meta_data / AICPU so 独立命名 /
  TND **累计** q-len 契约
- `op_src/` 与本仓 `op_src` 均只读；改动只落 `upstream/pr15336`（feat/fia-v2-sink 系分支）
- 上游跟踪：PR #15336 @34f94ca34 为 base；A5 移植开新分支，rebase 按 vendored README 对账
- **A3 行为不回退**：所有 SoC 差异必须走条件分支（ASCEND_COMPUTE_UNIT / __CCE_AICORE__ /
  容量常量参数化），改完必须重跑 910_93 全量门禁（七门禁 + C01–C26）
- kernel 缺陷只定位，走遗留通道；F2 本地盘构建纪律

## 5. 交付物

1. A5 分支 + 构建/门禁记录（`results/a5_port/`）
2. 迁移笔记（H1–H6 逐项裁决表）
3. `handoff/cannbot_handoff.json` 追加 a5_port 节（含 SoC/核数/容量参数与验证口径）
4. 门禁一致性表：910_93（回归）与 ascend950（新验）双列

## 6. 风险与开放问题

| # | 风险/问题 | 处置 |
|---|---|---|
| R1 | 9.1.0 工具链不支持 ascend950 编译目标 | 阻塞级：需申请 A5 CANN 工具链（P0 确认） |
| R2 | A5 硬件/仿真可得性未知 | 验收口径按 P2 分档，如实登记 |
| R3 | L0/L1/UB 容量与 910_93 不同 → 分块公式需按 A5 重推 | H2 公式已参数化，改常量即可 |
| R4 | A5 的 AICPU 子系统行为差异 | P0 确认 AICPU 形态存在性 |
| R5 | `dav_c220` 等核架构代号硬编码 | 跟随工具链，一般不需改 |

---

*交接完成。执行 session 从 P0 开始；每阶段完成后在本文件勾选并追加发现。*
