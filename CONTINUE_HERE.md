# FIA v2 Sink A3→A5 移植 — 跨机续作指引（CONTINUE HERE）

> 更新：2026-09-04（kw-13 终局）。本文档是另一台机器上恢复全部工作的单一入口。

## 0. 一页现状

**任务**：FIA v2 Sink 双算子（主算子 AICore + metadata AICPU 前置）从 A3(Ascend910_93) 移植到
A5(Ascend950PR)，兼容 A3（同一算子多 SoC，条件分支隔离），契约零变更。

**代码终态**：分支 `feat/fia-v2-sink-a5` @ `9faf61a2b`（= kw-9a + kw-9b + kw-10 + 钳位 + kw-11/12 hardening 探针 + kw-13 快照v2/双读/C2V2Last）
基线 `ded354d3d`（交接包 feat/fia-v2-sink）。战役全程 kw-1..10，证据链
`results/a5_port/MIGRATION_NOTES.md`（P0–P6.5）。

**验收口径**（user 两项裁决，见 results/a5_port/user_decision.md）：
A 路线 kernel 内修复经 kernel worker agent；ulp 口径收尾（G1/G2 bit + 其余 ≤2 bf16 ulp）。

**终态数据**（卡 4，基线=官方 torch_npu v2 API → 实际调度 aclnnFusedInferAttentionScoreV5）：

| 路径 | 精度 | 性能（sink/官方 wall-clock 中位） |
|---|---|---|
| GQA/MHA | G1/G2 **bit=True**；C21–C26 全 1–2 ulp；q=0 快检 56/56 | decode S1/S2 (kv 256K/512K) **0.855/0.834×（净赚 14–17%）**；prefill 大 M 1.3–10.6×（M-16 保守分块代价，遗留优化项） |
| MLA | **15/16 ulp**（C04–C09/C11–C20 全 0.0078=1 ulp）；C10 0.95 唯一残差 | 长 KV C12–C20 **0.107–0.592×（快官方 1.7–9.4×）**；新收敛 C05/C06/C07 0.72/0.55/0.34× |
| A3 回归 | 本机无 910_93 未实测；静态论证 A3 路径逐字节不变（全部改动 #if __NPU_ARCH__==3510 / 探针宏 / A5 条件 staging 内） | — |

**遗留清单**（按优先级）：
1. **C10 + C02 并列永久遗留（kw-11/12 确认同族）**：唯一稳定轴 = actMBaseSize<~64 ×
   并发核≥16 × 每 split ≥2 循环（mask/尾宽/B 数正交）；污染在 mm2 累计器；累计排除 11 类
   （nupdate 原子链 A/B、FD-combine 去混淆、mm2 M 链、L0C 握手、slot 复用等，见 P6–P6.7）；
   嫌疑终局收窄 = **fixpipe 原子写在 m<64 几何下写出错误终值**（V2 捕获值即错：race 已由
   C2V2Last 闭合但数值逐位不变，读时序/FD/除数/nupdate/L0C/slot 全排除）；下一战役 =
   dump 脏行完整 512 列累计器终值+除数 vs 宿主 fp64 模拟 n(i) 量化尺度链逐列对照，
   定位错误数值的代数形态（探针 [603..606] 惰性常驻，§P6.8）
2. bit 补齐（GQA ulp 差 → ProcessVec1Vf 移植）：dst strided store 布局两次探针未钉死，
   重启路径 = MIGRATION_NOTES P4 kw-7 节（attentionOutput 尾行单窗 dump 免拼装 / FlashMLA
   UB_VEC1_RES 注释反推）；VF 库接入三坑解法已固化（P4 kw-6 节）
3. 910_93 回归终验（回 A3 环境跑全量门禁）
4. GQA prefill 性能（M-16 分块 → mL0 级 staging，M>128 需扩展）
5. MLA 路径 V≠K 独立 value 寻址疑坏（absorbed 语义 V=K 不受影响）

## 1. 仓库结构（两层，同一远端）

```
远端 tyz: github.com/YzTongNiar/vllm-ascend.git   ← 凭据由持有人配置（勿写入任何文档）
├── 分支 feat/fia-v2-sink-a5   ← 代码仓（vllm-ascend/，含全部 kw 提交历史）
└── 分支 workspace-a5-port     ← 工作区仓（main = 上游 vllm-ascend master 镜像，勿动）（本目录：results/ docs/ tests/ pr15336/ build.sh 等）

本目录 = 工作区仓 checkout；vllm-ascend/ 是嵌套独立 git 仓（本仓 .gitignore 排除其内容）
```

**跨机恢复**：
```bash
git clone -b workspace-a5-port <remote> fia_sink_a5_port_base && cd fia_sink_a5_port_base
git clone -b feat/fia-v2-sink-a5 <remote> vllm-ascend   # 代码仓放回嵌套位置
```

## 2. 环境依赖

| 项 | 值 |
|---|---|
| CANN | 9.1.0（`source /usr/local/Ascend/ascend-toolkit/set_env.sh`；ascend950 工具链内置） |
| SoC 串 | 构建 `SOC_VERSION=ascend950pr_9599`（host_config 需具体型号串；短串 ascend950 过不了顶层 cmake） |
| python/torch | 3.11 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 |
| 设备 | Ascend950PR；**跑前 npu-smi 确认空闲卡**（本战役卡 2 中途被 vLLM 挤占污染过一批数据——教训：每批测试前查占用） |
| 构建 | `cd vllm-ascend && SOC_VERSION=ascend950pr_9599 MAX_JOBS=64 bash ../build.sh --fast`（全量 ~40min；A3 档默认 ascend910_9382 不变） |

## 3. 测试环境（每 shell）

```bash
cd <包>/a5_port_base_supplement/tests_pr15336    # 补充包已收入包内（含 4 处适配的门禁脚本 + 原值 case 数据）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>
export FIA_GATE_TREE=<包>/vllm-ascend
OPP=$FIA_GATE_TREE/vllm_ascend/_cann_ops_custom
export ASCEND_CUSTOM_OPP_PATH=$OPP:$OPP/vendors/custom_transformer   # 双条目缺一不可
python3 pr15336_gate.py mqa_c27   # 六门禁；MLA/全族走 results/a5_port/perf/perf_full_family.py --only <case>
```

## 4. 重编纪律（构建系统四坑——漏任何一步 = 编旧二进制白烧 20min）

改 kernel 源或编译选项后，必须**全部**执行：
```bash
cd vllm-ascend/csrc/build/binary/ascend950
rm -rf src/fused_infer_attention_score_v2_sink*              # 坑4: staging done flag 不跟踪源码修改
rm -f gen/fused_infer_attention_score_v2_sink*.done          # 坑3: stamp 与产物脱钩
rm -rf bin/fused_infer_attention_score_v2_sink
rm -rf gen/kernel_meta_FusedInferAttentionScoreV2Sink_*
cd <包> && SOC_VERSION=ascend950pr_9599 MAX_JOBS=64 bash build.sh --fast
```
另：`custom_compile_options.ini` 变更不触发重编（坑2）；`add_ops_compile_options` 放在
`BUILD_OPS_RTY_KERNEL` 分支内会被静默丢弃（坑1）；构建时 `TMPDIR` 指向大分区（/tmp 会被
torch_npu 残留撑满）。

## 5. 判读工具索引

- q=0 快检（数学正确性秒级判据）：`results/a5_port/probe_scripts/fia_round5_check.py`
- 全族精度+性能：`results/a5_port/perf/perf_full_family.py --only <case>`（逐 case 子进程隔离）
- kw-3..10 全部判别/探针脚本：`results/a5_port/probe_scripts/tmp_scripts/`（fia_kw3..fia_kw10）
- 数据：`results/a5_port/perf/{perf_a5_result.json, family_results_card4.txt}`
- 决策记录：`results/a5_port/user_decision.md`（A 路线 + ulp 口径两项裁决）
- 门禁双列终表与环境勘误：`results/a5_port/GATE_CONSISTENCY.md`
- 基线版本勘误：官方 torch v2 API 在 A5 实际调度 aclnnFusedInferAttentionScoreV5（同文件附 3）

## 6. 已定性勿重跑的假设（省 20min/轮 ×N）

P 传递错位 / SoftmaxFlashV2 整体失效（kw-1 误报，staging 坑所致撤回）/ scale / M 对齐 /
workspace 分段 / auto-sync / SSBUF / metadata 核数 / mm2 小 M 链（kw-10b 排除）——
均已实证排除，见 MIGRATION_NOTES 各节。
