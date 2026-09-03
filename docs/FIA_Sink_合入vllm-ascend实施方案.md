# FIA v2 Sink 双算子合入 vllm-ascend 实施方案（裁剪版设计，phase-1）

> 日期：2026-08-29　状态：设计完成，待实施（建议独立 session 领取 §9 任务分解）
> 参照：`docs/ascendC算子合入方案.md`（GMM 版方案，方向性文档）
> 实测仓：`/home/t00886357/fia_sink/vllm-ascend`（fia-sink-main，HEAD=8abbf2495）
> 本算子源：`/home/t00886357/npu_fused_infer_attention_score_v2_sink/op_src`（round_5 结档态）

---

## 1. 目标与现状

| 方案 | 内容 | 现状 |
|---|---|---|
| 方案二（现状） | **外挂态**：fia_v2_sink_ops wheel + 自研 OPP 包独立交付，`ASCEND_CUSTOM_OPP_PATH` 外部双条目；框架只做路由（commit 8abbf2495），半部署 engine 启动即 fail-fast | 已验证可用（round_2 整网多 batch + 4.6K 长生成） |
| **方案一（目标）** | **源码进仓**：双算子源码进入 `csrc/attention/`，随 `pip install vllm-ascend` 构建编译，OPP 产物随仓安装、运行时自举加载；删除外部 wheel/OPP 依赖 | 本方案设计 |

方案一收益（同 GMM 版论证）：源码与框架版本绑定、CI 可测、CANN 适配清晰、免手工拷贝。额外收益：框架路由 commit（8abbf2495）中"half-deployed fail-fast"与 `bootstrap_custom_op_env` 的外部依赖路径可一并收敛。

## 2. 侦察结论：三个硬骨头全部有在仓先例/机制

| 硬骨头 | 仓内解法 | 证据 |
|---|---|---|
| ① AscendC 工具链进外部构建 | 仓构建已内嵌 CANN `ascendc_kernel_cmake` + `ascendc_impl_build.py`；csrc 下已有十余个 AscendC 算子在编 | 根 `CMakeLists.txt`（ascendc_kernel_cmake 定位）、`csrc/CMakeLists.txt:528` |
| ② 双算子 + AICPU CpuKernel 打包 | **同源先例**：`csrc/attention/kv_quant_sparse_attn_sharedkv{,_metadata}`（含 `op_kernel_aicpu/` 与独立 CMakeLists）、`vllm_quant_lightning_indexer{,_metadata}` —— 与本算子同出 omni 谱系，结构同构 | `csrc/attention/` 目录清单 |
| ③ 运行时 OPP 路径自举 | 仓内已有 `vllm_ascend/utils.py::bootstrap_custom_op_env()`：把 `_cann_ops_custom/vendors/<vendor>` 前插 `ASCEND_CUSTOM_OPP_PATH`（可选 LD_LIBRARY_PATH）；envs.py 有集中 env 注册 | `utils.py:308-320`、`envs.py:88-96` |
| ④（附加）OPP 包布局 | `csrc/cmake/config.cmake:74`：`IMPL_INSTALL_DIR = packages/vendors/${VENDOR_NAME}_transformer/op_impl/...` —— 构建产物本身就是 `packages/vendors/<vendor>_transformer` OPP 布局，与我们的手工包同构 | config.cmake |

**结论：本任务 = 按仓内 dual-op 模板做一次源码移植 + 接线收敛，非新建构建体系。**

## 3. 目标结构

```text
vllm-ascend/
├── csrc/attention/
│   ├── fused_infer_attention_score_v2_sink/            # 主算子（仿 sharedkv 结构）
│   │   ├── op_host/          # def/infershape/tiling*(含 round_4 host 臂修复)
│   │   ├── op_kernel/        # AscendC 内核（含 round_5 L0B 分块修复、104 key 族）
│   │   ├── op_api/           # aclnn 两段接口（V3）
│   │   └── CMakeLists.txt
│   ├── fused_infer_attention_score_v2_sink_metadata/   # AICPU 前置
│   │   ├── op_kernel_aicpu/  # CpuKernel + 注册 json（libfia_v2_sink_aicpu_kernels.so 独立命名保留）
│   │   ├── op_api/
│   │   └── CMakeLists.txt    # 仿 kv_quant_sparse_attn_sharedkv_metadata
│   └── CMakeLists.txt        # 子目录挂载（沿用现有 SUB_DIR 循环）
├── vllm_ascend/
│   ├── utils.py              # bootstrap_custom_op_env（已存在，接新 vendor 目录）
│   ├── attention/attention_v1.py  # 路由已就绪(8abbf2495)，改从仓内包解析
│   └── envs.py               # VLLM_ASCEND_ENABLE_FIA_V2_SINK(_GRAPH) 已注册
└── tests/ops/                # CPU 单测已入(8abbf2495)；设备门禁迁移 §7
```

## 4. 源码移植映射（必须携带 round_4/5 修复）

| op_src 来源 | 去向 | 必带修复/注意 |
|---|---|---|
| `fused_infer_attention_score_v2_sink/op_host/*` | csrc/attention/fused_infer_attention_score_v2_sink/op_host/ | **CheckGqaDSupport (192,0,192) 臂**（round_4） |
| `.../op_kernel/*` | 同上 op_kernel/ | **104 GENERAL key 族全矩阵**（round_4）；**ComputeMm2 K 分块按 L0B 容量推导**（round_5）；命名保持 `_v2_sink` 后缀 |
| `.../op_api/*` | 同上 op_api/ | schema 契约零变更（30 输入/4 输出/19 属性） |
| `_metadata/op_kernel_aicpu/*` | csrc/attention/..._metadata/op_kernel_aicpu/ | **AICPU so 独立命名 libfia_v2_sink_aicpu_kernels.so 保留**；F3 归一修复在内 |
| `_metadata/op_api/*`、`op_graph/`、`config.ini` | 对应目录 | |
| `attention/common/`（offset/memcopy/seq 解析器/split_core） | csrc/attention/common/（已有 common/，做合并冲突检查——sharedkv 系可能已带同名文件） | **重点排查头文件同名冲突与版本差** |
| vendored 声明 | `csrc/attention/fused_infer_attention_score_v2_sink/README.md` | Upstream: omni-ops ai_infra_fused_infer_attention_sink(+_metadata) @ 本仓 pin；Local mods: 全局改名/SoC 910_93/104 key 族/CheckGqaDSupport (192,0,192)/L0B 容量推导/F3 归一（plans/03 改表为底） |

## 5. 运行时接线设计

1. 构建安装后，OPP 产物落 `${VENDOR_NAME}_transformer` 布局 → 安装步骤拷入 `vllm_ascend/_cann_ops_custom/packages`（与 sharedkv 系同机制，实施时确认 `_CUSTOM_OP_BASE_DIR` 的填充点）。
2. `bootstrap_custom_op_env()` 已实现前插逻辑；确认其调用时机早于 torch_npu 初始化（现有 dual-op 已依赖它，预期无需新写）。
3. `attention_v1.py` 的 fail-fast 从"外部 wheel+OPP"改为"仓内包"判定；`VLLM_ASCEND_ENABLE_FIA_V2_SINK(_GRAPH)` 语义不变（默认 0，F1 风险注记保留）。
4. envs.py 中 "NOT run-verified yet (phase A)" 注释已过时（round_2 整网已验证多 batch+4.6K 长生成），实施时一并更新。

## 6. 构建与 CI

- **编译时长**：本算子 TILINGKEY_PAR_COMPILE 共 258 个 key 变体（103+105+104），单机 ~100s@640 核；CI 核数少时显著变长 → 选项：(a) 接受；(b) 裁剪未验证形态（GQA 仅保留 TND/BNSD q 布局 + PA/NonPA，非量化）——**建议 CI 默认裁剪集 + 本地全量**，裁剪集清单在方案评审时定。
- **CANN 版本耦合**：csrc 构建依赖 CANN（ascendc_kernel_cmake/opc）；按仓现状（torch 2.10.0 硬校验先例）锁定 CANN 9.1/9.2 双验证。
- **F2 对应**：CI 构建天然本地盘；本地验证延续 /tmp 纪律。

## 7. 测试与验收

| 层 | 内容 | 来源 |
|---|---|---|
| CPU 单测 | 已有（8abbf2495：开关默认/init 门禁/路由条件/seq 构造/meta 缓存） | 保留+扩展 |
| 设备门禁 | C01（MTP_verify MLA 形态）、C21–C26+mqa_c27/c28（GQA）、B1 kvlen sweep 至 32K、B=2/B=4 累计输入矩阵 | 从 `results/round_5_fix2/` 脚本迁移，指向仓内构建产物 |
| 整网 | GLM-5.2+DSpark：多 batch 并发 + 4.6K 长生成 + （新）32K 长生成 | round_2 验证口径 |

验收标准：方案一构建的包与 round_5 交付包在上述门禁上**结果一致**（bit/门限同口径）；`ASCEND_CUSTOM_OPP_PATH` 外部条目清空后功能不回退。

## 8. 风险与开放问题

| # | 风险/问题 | 处置 |
|---|---|---|
| R1 | `attention/common/` 头文件与 sharedkv 系同名冲突 | 移植前 diff common/ 两树；必要时按算子子目录隔离 include |
| R2 | VENDOR_NAME 命名（config.cmake `${VENDOR_NAME}_transformer`）与真 omni 包同装冲突 | 为 vllm-ascend 选独立 vendor 名（如 vllm_ascend_custom），bootstrap 同步 |
| R3 | CI 编译时长（258 keys） | 裁剪集策略（§6） |
| R4 | AICPU so 装载（DNN_VM_AICPU 自定义 so 在仓包内是否被 runtime 接受） | sharedkv_metadata 已在仓内运行 → 有先例；实施首轮即冒烟验证 |
| R5 | 两仓源码漂移（本仓 op_src 为权威源，vllm-ascend 为发布载体） | 建立单向同步脚本（op_src → csrc）+ 版本戳，防双头改 |
| R6 | `_CUSTOM_OP_BASE_DIR` 填充机制未完全确认 | 实施首任务：读 utils.py 全文 + sharedkv 的安装/装载链 |

## 9. 任务分解（供实施 session 领取）

1. 读 `utils.py`/`bootstrap_custom_op_env`/`_CUSTOM_OP_BASE_DIR` 与 sharedkv 双算子的构建-安装-装载全链（R6 清零）。
2. 移植双算子源码进 `csrc/attention/`（§4 映射表 + round_4/5 修复清单核对）。
3. CMake 挂载 + 独立 vendor 名（R2）+ 本地构建冒烟（opc 编译通过、包布局正确）。
4. bootstrap 接线 + attention_v1 fail-fast 改造 + envs.py 注释更新（§5）。
5. 设备门禁迁移与全量回归（§7），对照 round_5 结果一致性。
6. CPU 单测扩展 + vendored README + （可选）CI 裁剪集。
7. 移除外部 wheel/OPP 依赖路径，更新 `handoff/` 与本文档状态。

---

*phase-1 产出（本文件）；实施建议新 session 按上表领取，上下文 briefing = 本文件 + `docs/ascendC算子合入方案.md` + `handoff/cannbot_handoff.json`（round=2）。*
