# FIA v2 Sink 算子 A3→A5 移植基座交接包（2026-09-01）

本包 = 在新环境启动 **A3(Ascend910_93)→A5(Ascend950) 移植任务** 的全部材料：
算子源码（方案一：vllm-ascend csrc 内置形态）+ 交接文档 + 门禁。

## 0. 先读这两份

1. **`docs/FIA_Sink_A3到A5移植交接.md`** —— 移植任务总 briefing：
   已勘察事实、A3 假设清单 H1–H6、分阶段计划 P0–P3、铁律、风险表。**从 P0 开始执行。**
2. `docs/FIA_v2_Sink_算子接口与实现说明.md` —— 算子接口/参数/支持矩阵/契约/调用方法。

## 1. 内容地图

| 路径 | 内容 |
|---|---|
| `vllm-ascend/` | **完整 clone**（含 .git 与分支历史）。权威分支 `feat/fia-v2-sink` @ `ded354d3d`，双算子位于 `csrc/attention/fused_infer_attention_score_v2_sink{,_metadata,_common}/`，python wrapper 在 `vllm_ascend/ops/fia_v2_sink.py` |
| `docs/` | 交接文档（上）+ 算子接口与实现说明 |
| `tests/pr15336/` | 设备门禁与探针（`FIA_GATE_TREE` 环境变量可配） |
| `AUTHORITY.md` / `build.sh` | 权威声明 / 独立构建壳 |

已验证状态：`ded354d3d` = opsproto 注册修复（infershape 显式收集）+ 内部命名中性化，
其父提交链含 round_4/5 全部修复（104 tiling key 族、(192,0,192) 路由臂、L0B 容量分块）。

## 2. 快速开始（A3 基线复现，验证环境可用）

```bash
cd vllm-ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh     # CANN 9.1.0
export SOC_VERSION=ascend910_9382 MAX_JOBS=64 COMPILE_CUSTOM_KERNELS=1
python3 setup.py build_ext --inplace                    # ~20-60min（fatbin 长编译为长杆）
# 门禁（Ascend910_93 设备，卡 8–15 空闲段）：
cd /home/t00886357/npu_fused_infer_attention_score_v2_sink
source deploy_env.sh 15
python3 tests/pr15336/pr15336_gate.py mqa_c27           # 须 bit_exact=True（其余同理）
```

⚠️ 构建缓存含绝对路径：**换环境/换目录后必须清 `csrc/build` 与 `build/` 再编**（陈旧缓存
会导致 ld.lld 找不到中间产物）。并行 opc 有非确定性竞态前科（双 fatbin 变体并发时
kernel_meta 中间目录竞态）——复现则降并行或串行化 kernel 阶段。

## 3. 移植任务入口

按 `docs/FIA_Sink_A3到A5移植交接.md` 执行：
P0 勘察（ascend950 工具链支持确认 / A5 规格）→ P1 编译打通 → P2 门禁迁移 → P3 收尾。
铁律：契约四红线、A3 行为不回退（SoC 差异走条件分支）、TND 累计 q-len、op_src/存档只读。
