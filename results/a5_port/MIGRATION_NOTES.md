# FIA v2 Sink A3→A5 移植笔记（a5_port）

任务：FIA v2 Sink 双算子 A3(Ascend910_93)→A5(Ascend950) 移植。基线 `feat/fia-v2-sink` @ `ded354d3d`。
本文件为活文档：H1–H6 逐项裁决表 + 各阶段记录。

## P0 环境与规格勘察（2026-09-02）

### 工具链（R1 裁决：解除）

| 项 | 值 |
|---|---|
| 唯一 CANN | `/usr/local/Ascend/cann-9.1.0`（ascend-toolkit → cann 同源；9.2.0 已不在本机） |
| ascend950 编译支持 | ✅ `ascendc_kernel_cmake` 含 ascend950（const_var.py / config.cmake ASC_VALID_SOC_LIST） |
| 编译 SoC 串 | `ascend950` → `Ascend950PR_9599`（const_var.py SOC_MAP_EXT） |
| AICPU 工具链 | `aicpu-config.cmake` 存在（R4 解除） |
| CPU 仿真 | dav-3510 在 simulator 映射表（可作无设备调试备选） |
| python | 3.11.10 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 |

### 设备（R2 裁决：bit 口径可行）

8× Ascend950PR（npu-smi 25.7.rc1），卡 2/3 空闲（任务指定优先），0/1/4-7 被 vLLM 占用。
运行态型号 Ascend950PR_9579。

### A5 规格 vs A3（H1/H2 裁决依据）

来源：`cann-9.1.0/x86_64-linux/data/platform_config/{Ascend950PR_9579,Ascend910_9391}.ini`

| 项 | A3 9391 | A5 9579 | 裁决 |
|---|---|---|---|
| cube(AIC) 核 | 24（ini；H1 声明实跑 36） | **28**（ini + torch_npu 双证） | H1：28 ≤ 36 表上限，安全 |
| vector(AIV) 核 | 48（ini；H1 声明 72） | **56** | H1：56 ≤ 72 表上限，安全 |
| 核架构 | dav-**c220** | dav-**c310** | 代际升级；`NpuArch=3510` → `__NPU_ARCH__==3510` |
| 核型 | split | split（同） | 无形态差异（初判的"MIX 风险"不成立） |
| L0A / L0B | 64K / 64K | 64K / 64K | **相同** → H2 kCap 参数化无需改值 |
| L0C | 128K | **256K** | 翻倍，向上安全 |
| L1 | 512K | **512K** | **相同** → H2 L1 分区 exact-fit 常量无需改 |
| UB | 192K | **248K** | 变大，向上安全 |
| L2 | — | 128M | 记录 |
| AICPU | 有 | ai_cpu_cnt=7 | R4 解除 |
| vec_calc_size | 128 | 128 | 同 |

**H2 结论**：L1/L0A/L0B 三项关键容量 A5 与 A3 完全一致，L0C/UB 只增不减 →
现有分块公式与 L1 分区常量**预期可直接复用**，无需矩阵化重推。编译期由门禁复核。

### arch 代号（H5 裁决；修正交接文档 §1.3 的笔误）

权威：`csrc/CMakeLists.txt` `SOC_VERSION_LIST`/`ARCH_DIRECTORY_LIST` 逐位对应：
ascend310p→arch22, ascend910b→arch32, ascend910_93→**arch32**, ascend950→**arch35**, kirinx90→arch32。
**A5 = arch35**（910_93 = arch32，交接文档"arch35=910_93"系笔误）。FlashMLA 先例佐证。

### 上游 A5 先例（基线祖先链内，e018b4023，2026-08-31）

`feat(attention): use FlashMLA tiling sink on A5` —— FlashMLA + metadata 双算子 A5 适配全模式：
- `op_kernel/arch35/` 子目录（`*_arch35.h`）+ `COMPUTE_UNIT Ascend950PR_9599` 编译选项 +
  `-DENABLE_CV_COMM_VIA_SSBUF=true`（A5 cube↔vector 跨核通信）+ `add_modules_sources_with_soc`
- **顶层 CMakeLists.txt 显式将 FIA v2 sink 限 A2/A3**：ascend950 分支不编译 FIA binding、
  不定义 `VLLM_ASCEND_ENABLE_FIA_V2_SINK`（改定义 `VLLM_ASCEND_ENABLE_FLASH_MLA`）
- `csrc/CMakeLists.txt`：ascend950 时 `ENABLE_AICPU ON` 仅当 op 表含 `flash_mla_with_kvcache_metadata`
- 本任务 = 解除该限制：FIA sink 算子级 CMake 无 SoC 门（通配扫描全 SoC），改动集中上述两处 + build.sh

### H1–H6 裁决表（P0 初判 → P1 编译期 → P2 运行期逐项落锤）

| # | 内容 | P0 初判 | 状态 |
|---|---|---|---|
| H1 | metadata 表维度 aic[36]/aiv[72] | A5 28/56 在上限内；wrapper 从 `get_stream_limit()` 动态取 cube/vector，A5 属性名兼容 | P0 定：不改 |
| H2 | L1 分区 exact-fit 常量 | L1=512K/L0A=64K/L0B=64K 与 A3 全同；L0C/UB 只增 | P0 定：不改，门禁复核 |
| H3 | `__CCE_AICORE__==310` 分支 | A5=dav-c310；宏实际值待编译期 dump（`__NPU_ARCH__==3510` 已确认） | P1 验证 |
| H4 | dav_c220 include 路径 | 跟工具链走（A5 用 dav-c310 头） | P1 编译验证 |
| H5 | arch 子目录约定 | arch35=A5 已定；FIA 扁平布局保持（kernel 源若直接可编则不拆目录，小决策） | P1 定 |
| H6 | 指令组合（load2d/mmad/fixpipe） | 编译通过≠正确，P2 门禁验证 | P2 |

### 门禁缺口（P2 前置，交接包未随附）

- `gqa_case_builder.py`（bits_equal/build_inputs/parse_case/resolve）、`case_builder.py`
  （load_cases + C01–C26 数据）在原工作区 `tests/`，原工作区已不存在 → P2 自包含重建
  （`gate_kvlen`/`gate_b2` 已内联自包含，可直接用）
- 门禁脚本硬编码 `PR_TREE`/`OURS`，**不读** `FIA_GATE_TREE`（交接文档"可配"与实现不符）→ P2 适配

### 风险表更新

| # | 原判 | P0 后 |
|---|---|---|
| R1 | 9.1.0 不支持 ascend950？ | **解除**（支持） |
| R2 | A5 设备可得性 | **解除**（卡 2/3 空闲，bit 口径） |
| R3 | 容量不同需重推分块 | **基本解除**（L1/L0A/L0B 同，仅 L0C/UB 增） |
| R4 | AICPU 形态 | **解除**（ai_cpu_cnt=7） |
| R5 | dav 代号硬编码 | 跟工具链，P1 编译验证 |

## P1 编译打通（2026-09-02）

### 改动清单（4 处，全部落 `feat/fia-v2-sink-a5` 分支）

| # | 文件 | 改动 | 理由 |
|---|---|---|---|
| 1 | `csrc/attention/…_sink/op_kernel/fused_infer_attention_score_v2_sink_metadata.h` | `FIA_MAX_AIC_CORE_NUM` 26 → 36（对齐 AIC_CORE_NUM 表上限，附注释） | **H1 kernel 侧缺口（交接文档未覆盖）**：MLA FlashDecode 的 FD 头本地数组容量 26 = A3 实跑 24 核+2 余量；A5 28 核（AIC 侧 28>26、AIV 侧 56>52）越界写 UB 栈。A3 侧仅容量变大、循环上界仍由 metadata usedCoreNum 决定（≤24），行为零变化 |
| 2 | 顶层 `CMakeLists.txt` | ascend950 分支 VLLM_ASCEND_SRC 增 `csrc/attention/…_sink/torch_binding/*.cpp` | 上游 e018b4023 把 FIA binding 限 A2/A3；本任务解除（A5 也编 FIA binding） |
| 3 | 顶层 `CMakeLists.txt` | ascend950 分支宏改 `VLLM_ASCEND_ENABLE_FLASH_MLA` + `VLLM_ASCEND_ENABLE_FIA_V2_SINK` 双定义 | torch_binding.cpp:2948 的 `#ifdef` 注册块需要 |
| 4 | `build.sh` | `SOC_VERSION` 环境变量化（默认 ascend910_9382 不变；A5 档 `SOC_VERSION=ascend950`）+ 产物 ls 目录按 SoC 分支 | 任务书 P1 要求；A3 默认行为不变 |

**不需要改的（审查结论）**：
- `csrc/CMakeLists.txt` ENABLE_AICPU 条件：FIA metadata 走 `add_modules_sources_aicpu` 宏，
  宏内自动 `set(ENABLE_AICPU ON CACHE FORCE)`（obj_func.cmake:468），CMakeCache 已验证 ON
- 算子级 CMakeLists：attention/ 通配扫描无 SoC 门；FIA sink 三算子自动入编（950 全算子编译）
- metadata AICPU so 独立命名：install RENAME 契约段原样保留
- L1/L0 分区常量（H2）：A3/A5 L1 均 512K、L0A/B 均 64K；GQA 路径 L1 总用量 ~464K<512K、
  MLA 72K 块 × 缓冲 <512K、L0C 用 128K≤256K——exact-fit 保持，**零改动**
- H4：算子源码无 dav_c* 硬编码 include（框架头由构建系统注入）
- python wrapper `fia_v2_sink.py`：无 SoC 路由；核数取 `get_stream_limit()` cube/vector（A5 属性兼容）

### H1 补充发现（重要）

A3 的 `get_stream_limit()["cube_core_num"]` 实际返回 **24**（与 9391 ini cube_core_cnt=24 一致）；
交接文档 H1 说的"AIC36/AIV72"是 metadata 表的**合同容量上限**（1024 定长契约的组成，
static_assert 保护），非 A3 实跑核数。wrapper 每次显式传核数，schema 默认 24/48 不被使用。
A5 实跑 28/56（torch_npu properties 双证）。表容量 36/72 对 A5 足够，**不改**。

### 构建验证（build_a5_01 → 09，四轮排障）

| 轮 | SOC 入参 | 结果 | 根因/处置 |
|---|---|---|---|
| 01 | ascend950 | 顶层 cmake configure 挂 | host_config.cmake 支持列表只含**具体型号**串（ascend950pr_9599 等），不含短串；改传 ascend950pr_9599（build_aclnn 的 `^ascend950` 正则两态兼容） |
| 02 | ascend950pr_9599 | ld multiple definition | **A5 分支 CUSTOM_OPS 未列 FIA**（已补）+ 上游 FlashMLA 引入的 `common/op_host/fia_tiling_shape.cpp` 与 FIA vendored 修复版（222 行漂移）同符号双编进聚合 tiling obj → A5 列表去掉 flash_mla 两算子（本树交付 FIA；FlashMLA 归上游交付） |
| 03 | 同上 | symbol.cmake 引用已移除的 flash_mla 目标 | op 列表变更 = 配置变更，陈旧缓存（F2 纪律）→ 全清重跑 |
| 04 | 同上 | ✅ BUILD_EXIT=0 | 全量零错误；OPP 布局 kernel/ascend950/ + config json + libcust_opapi/opsproto/optiling + AICPU so（AArch64 交叉编译正确）+ x86_64 torch 扩展 |
| 05-09 | 同上 | P2 修错循环 | 见 P2 节（SSBUF 选项 + 构建系统三坑） |

**构建系统三坑（P1 附带发现，上游回赠候选）**：
1. `add_ops_compile_options` 放在 `if (BUILD_OPS_RTY_KERNEL)` 分支内会被静默丢弃（本仓 open-project
   构建恒 OFF）——FIA 原模板的 `--cce-auto-sync=off` 从未生效（A3/A5 基线一致，非行为差异）；
   已移到无条件调用位（对齐 CausalConv1d 模式）
2. `custom_compile_options.ini` 变更**不触发** kernel 重编（ninja 无依赖跟踪）
3. kernel 产物用 `gen/*.done` stamp；删 bin 输出不清 stamp → "not any obj compile success"。
   正确重编法：`rm gen/<op>_ascend950_*.done` 后 ninja 重跑

## P2 功能等价验证（2026-09-02，进行中）

### 门禁执行环境

- 设备：卡 2（Ascend950PR_9579）；`ASCEND_RT_VISIBLE_DEVICES=2`
- 被测树：`FIA_GATE_TREE=fia_sink_a5_port_base/vllm-ascend`（A5 构建）
- OPP 双条目：`_cann_ops_custom:_cann_ops_custom/vendors/custom_transformer`
- 门禁脚本：补充包 `a5_port_base_supplement/tests_pr15336/`（自包含版，替代 v1 交接包缺 helper
  的门禁——含原值 gqa_case_builder/case_builder/fia_v2_cases.csv）
- 补充包脚本适配 2 处（非算子逻辑）：metadata 直调参数名 qlen/kvlen→lengths、主算子
  metadata→meta_data（schema 形参名归一，均在包装函数内单点转换）

### 证据链（q=0 受控实验系列）

1. 双算子在 A5 **链路全通**：metadata AICPU 计算正确（G1 解码：M_BASE=512 S_INNER=20
   USED_CORE=8 bn2End[1..8] 递增，与接口文档 §4.1.5 语义一致）；主算子无崩溃有输出
2. G1 bit 门禁 fail：mere=1.0（首轮）→ 加 SSBUF 后 0.287（有改善非根因）；
   D128（G2）同 fail、C21 mere=1.8 —— 全形态错，根因通用
3. **法医定位**：
   - 输出结构正确（与 ref 逐位 cos 0.993，非重排/非搬运失败）
   - q=0 受控实验：官方 out=V 均值（距 0.0017 数学正确）；**我们 out 距 V 均值 0.27**
     → softmax 权重非均匀 → P 矩阵到达 vector 侧时已被污染（q=0 时 P 应恒 0）
   - lse(q=0) 矩阵：期望全 log(20)=2.996；实际仅尾部 (t,h) 正确（T=7 时 5/56 全在 t=6；
     T=16 时 62/128 尾部）—— **结构性部分正确 = 竞态特征**
   - scale 假设排除（多 scale 拟合不收敛）；M 对齐假设排除（T=8 全错）
   - host tiling 日志实锤：`block dim: 28 aiv: 56 aic: 28`，runtime `coreDim=28 taskRation=2`
     —— blockDim=coreNum=28，**workspace 分段错位假设排除**
   - metadata 核数 24/48（补充包直调默认）→ 28/56 有改善（max_abs 2.0→0.89）但非根因；
     正式调用路径（wrapper get_stream_limit）自动传对
4. **当前根因结论**：主算子 kernel 的 cube→vector 数据传递时序在 dav-c310 上存在
   跨 pipe/跨核事件链缺口（fixpipe→GM→vector 读的同步），H6（指令组合/时序）范畴的
   深层架构差异；SSBUF/auto-sync 等编译选项能改变（缓解）结果佐证时序性
5. 待验证：`--cce-auto-sync=on`（build_a5_10）——编译器保守同步注入

### P2 最终结果（auto-sync + SSBUF 版，build_a5_10 产物）

| 门禁 | 结果 | 备注 |
|---|---|---|
| G1 mqa_c27 (D192 MHA g1) | bit_exact=False mere=0.268 | 选项演化 mere：1.0 → 0.287(+SSBUF) → 0.268(+auto-sync) |
| G2 mqa_c28 (D128) | bit_exact=False mere=1.0 | |
| G3 C21 (GQA 变长 B8) | bit_exact=False mere=1.81 | |
| G4 kvlen_4096 | bit_exact=False mere=0.83 | 写偏移假设检验：长序列未改善，假设未获支持 |
| G4 kvlen_32768 | bit_exact=False mere=0.85 | |
| G5 b2_248 | bit_exact_vs_concat=False mere=0.57 | |
| G6 C01 MLA | **PASS** out=(64,16,512) finite=True | 可运行+有限口径通过 |

**auto-sync 结论**：仅微扰（0.287→0.268），编译器同步注入非修复；错误确定性
（重跑 mere 稳定）排除竞态 → 确定性数据路径错（P 传递/格式层）。

**P2 定案**：链路全通 + metadata 正确 + MLA 可运行，GQA bit 门禁未过；
根因定位主算子 cube→vector P 矩阵路径（dav-c310 适配层），按 §7 阻塞决策登记
（a5_port_feedback.md），修复选项 A/B/C 待确认。

### C01 门禁脚本适配记录（补充包 4 处，非算子逻辑）

1. metadata 直调参数名 qlen/kvlen→lengths 归一（gate.py 包装函数单点转换）
2. 主算子直调 metadata→meta_data 归一（同上）
3. C01 重复传参 bug（query_rope/key_rope 重复）
4. C01 改 ExtensionFileLoader 直载（import vllm_ascend 完整包需 wheel 安装产物）

## P4 kernel-worker 修复轮（2026-09-02，dump 实证驱动的根因定位）

任务：修 GQA P 矩阵传递错位（user_decision 选 A 路线）。5 轮构建（build_a5_round1..5.log），
未过 G1；根因已从"P 传递"修正定位到 **vector 侧 SoftmaxFlashV2 库调用在 dav-c310 上失效**。
以下为证据链与已合入/未合入改动。

### 证据链（metadata 通道 dump，round5 定稿）

dump 通道演化：round1-4 把探针写 attentionOut——**全部被 Vec2 输出路径覆写，读数无效**
（教训：宿主可见 dump 不能用会被生产写满的 buffer）。round5 改道 metadata 张量尾区
（int32[586..1023]，AIC/AIV/base 表之外），q=0 受控实验（P 应恒 0）：

| 探针 | 位置 | 结果 |
|---|---|---|
| D' L0C@fixpipe | AIC 首块 L0C 经生产同款 fixpipe 直写 | **全 0（正确）** |
| B' slot@Vec2 | Vec2 回读 mm1Res GM slot | **全 0（正确）** |
| A' P@Vec1 | vector DataCopy 后、softmax 前 | **全 0（正确）** |
| lse | 输出 | **±e35/NaN 垃圾（0/56 正确）** |

同进程三连跑 lse 逐位相同（1.61e+35/1.54e+35/...）→ 确定性、非竞态、非 workspace 脏读
（若早读脏内存，第二次运行 workspace 已被第一次写成 0，结果应变）。
q=0 时所有任务 P=0，竞态也读不出非零 lse → **P 全链路（CopyQToL1 ND2NZ → load → mmad →
L0C → fixpipe → GM → vector 读）在 P2 基线同步下已被证实正确**；病灶在 A' 采样点之后的
vector 内部计算。

### 根因（行级）

`op_kernel/fia_block_vec_nonquant_sink.h:709/720`：
`SoftmaxFlashV2<SOFTMAX_TYPE, true, true, false, false, FIA_SOFTMAX_FLASHV2_CFG>`（AscendC
高层 softmax 库，isCheckTiling=false 绕过校验 + SoftMaxFlashV2TilingFunc 自算 tiling +
tmpBuff1 32KB workspace）在 dav-c310 上从干净输入产生确定性垃圾。旁证：本仓 A5 已验证的
FlashMLA arch35 vec **不用该库**——softmax 为手写 VF 族（Sub/Exp/Max + ResetSoftmaxBuffer +
ComputeLseOutputVF + arch35 专属 DataCopySoftmaxLse*Arch35 拷出函数）；官方 vendored
fused_infer_attention_score/arch35 GQA vec 同样不见 SoftmaxFlashV2 调用。c220 上该库调用
工作正常（A3 六门禁全绿），属 dav 代际差异（c310 向量指令/库 tiling 假设不同）。

### 已排除（本轮实证，勿重跑）

- fixpipe 参数（V220/C310）：MLA 路径 V220 原样参数在 A5 通过（C01），fixpipe 机制无罪
- 跨核同步（mode-2 vs mode-4/intra-block + 显式消费队列 pipe）：回退到 P2 基线后 P 链路正确；
  mode-4 改造（round2，已回退）未改变 q=0 结果特征
- L1→L0A/L0B 装载：round4 换 matmul.h 310 分支装载器（MatmulKPP，MLA 同款）后输出与
  load3d 版逐位相同——装载层非当时瓶颈（KPP 版已保留合入，见下）
- load3d 手搓路径（LoadAToL0/LoadBTransposeToL0/LoadBToL0 else 分支）：matmul.h 在
  `__CCE_AICORE__==310` 分支整体切换到 LoadData2D 族——手搓 load3d 在 c310 无成功先例，
  虽然 round5 证明其结果与 KPP 相同，仍按官方框架对齐替换（A5 门控）
- workspace 脏读/竞态：三连跑实验排除
- Q GM 寻址（CopyQueryGmToL1/OffsetCalculator）：D' 全零证明 Q 正确到达 L0C

### 已合入改动（feat/fia-v2-sink-a5，全部 A5 门控/编译外，A3 路径零变化）

1. `fia_block_cube_nonquant_gqa_sink.h`：A5 下 mm1/mm2 内层 L0 装载改走
   `AiInfraInferenceCommonFaBaseMatmul::MatmulKPP`（+BufferManager/BuffersPolicyDB 管理
   L0A/L0B，与 MLA 路径同款；A3 走原 #else load3d 分支不变）。限制：A5 暂按 mL0≤128 整块
   装载（门禁族 gS1≤128 均覆盖；M>128 的 mL1 复用需 mL0 级 staging，未放开）
2. 同文件：FIA_A5_DEBUG_DUMP 宏控的 metadata 尾区探针（D'/A'/B'，默认编译外）
3. `fia_block_vec_nonquant_sink.h` / `fia_kernel_nonquant_sink.h`：同宏控探针与注入
4. `fia_public_define.h`：FIA_CROSS_SYNC_MODE/FIA_CROSS_AIV1_OFFSET 常量（当前未引用，留作
   同步改造复用）

### P4 续（kw-2 轮，2026-09-02 下午）：**staging 陷阱推翻前 7 轮全部实验 + 新 kernel 死锁未解**

**构建系统第 4 坑（比 P1 三坑更隐蔽，上游回赠候选第一优先级）**：
`csrc/cmake/func.cmake add_ops_src_copy` 的 staging 以
`binary/ascend950/src/<op>/<op>_ascend950_src_copy.done` 为 custom_command OUTPUT——
**只在 flag 缺失时拷贝源码，之后 op_kernel 的任何修改都不触发重暂存**。正确重编纪律
（P1 纪律的修正版）：
```
rm -rf csrc/build/binary/ascend950/src/<op>          # ← 新增：staged 源树 + src_copy.done
rm -f  csrc/build/binary/ascend950/gen/<op>*.done
rm -rf csrc/build/binary/ascend950/bin/<op>
rm -rf csrc/build/binary/ascend950/gen/kernel_meta_<OP大写>_*   # cce 中间物
```
**后果**：kw-1 的 5 轮构建（round1-5）与 kw-2 的 round6/7 全部编译的是 05:50 的初版
staged 源——所有"探针读数"（包括 round5 的 D'/A'/B' 全零）实为探针从未编入二进制，
**P4 主体中"round5 dump 证实 P 链路正确、SoftmaxFlashV2 失效"的根因结论作废**；
各轮"输出逐位相同"的现象由此得到平凡解释（同一二进制）。仍然成立的原始事实：
lse=±e35/NaN、输出与 q 无关、同进程三连跑逐位相同（这些都是**原始 A5 二进制**的行为）。

**kw-2 实际完成的工作**（staging 修复后 round8d 才真正编入）：
1. A5 门控手写在线 softmax（`fia_block_vec_nonquant_sink.h SoftmaxFlashV2Compute`
   `#if==3510` 分支）：RowMaxForLongColumnCount/RowSumForLongColumnCount + 行广播 Sub
   （镜像 RowMuls repeat 参数）+ Exp/Mul/Add 折叠，输出槽位布局与原契约一致（BRC/非BRC
   均实现）；ProcessVec1SingleBuf 的 A5 mSplitSize 减半（tmpBuff1 需同时容纳 P 副本
   ≤4096 与行级 scratch）。
2. round7 探针改道 metadata 尾区（int32[592..1023]）：A'=P pre-Muls、A''=post-Muls、
   E=VEC 自检（Duplicate/Muls/Add 后应全 1.0）、D'=L0C@fixpipe（+960）。
3. round8d（首个真正编入修改的构建）结果：**kernel 死锁**（q=0 探针 90s 超时无任何
   输出，无 plog）。改动面=KPP 装载(kw-1)+手写 softmax+探针三者叠加，未定位到具体
   卡点。首要嫌疑：(a) KPP Buffer 框架的事件与 GQA AllocEventID 的 L0AB_EVENT0/1
   预置互相干扰（MLA 的 AllocEventID 只预置 L1 事件）；(b) 手写 softmax 的
   RowMax/WholeReduceMax 在 c310 上的约束；(c) AIV/CIC 跨核 wait（P2 mode2）在 KPP
   时序下不满足。

### P4 终（kw-3 第一刀，2026-09-02 傍晚）：**首批真实读数完成方向裁决**

构建：kw3a（纯探针版 = 原 kernel 路径 + A'/A''/E/D' 四探针，KPP 与手写 softmax 均已
回退）。q=0 与 q=8 双输入（T7/H8/D192/kv20，C27 形态），读数（metadata 尾区通道，
此次探针真实编入——staged src 已按修正纪律清理）：

| 探针 | q=0 | q=8 | 判读 |
|---|---|---|---|
| D' L0C@fixpipe | ±2.07e9（40/64 非零） | ±2.07e9（值不同） | **mm1 结果 = 垃圾但 q 敏感** |
| A' P@Vec1 读 | **与 D'(q=0) 逐位相同** | 与 D'(q=8) 同量级 | **P 传递路径位精确正确** |
| A'' post-Muls | ≈A' 混合值 | 同 | 读法有 V→MTE3 竞态，本轮无效 |
| E VEC 自检 | 垃圾且随 q 变 | 常量垃圾 | 同上，本轮无效 |
| out | e34（q=0 与 q=8 逐位同） | 同 | exp(2e9×0.072)→inf 饱和，完美解释 |

**裁决（三层）**：
1. **P 传递路径完全正确**（A'==D' 逐位）——P2 的"P 矩阵传递错位"方向彻底终结。
2. **病灶 = cube 的 Q→L0A 装载（mm1 A 操作数）**：P(q=0) 应恒 0，实测 2.07e9 且
   q 敏感 → load3d 版 LoadAToL0 在 dav-c310 读错位 L1（混入真实 Q 与垃圾），与
   coordinator 增量洞察一致。量级佐证：正常 q·k·scale ≤ ~55，实测 e9（错位读取）。
3. **vector 侧大概率无辜**：P=2e9 → exp 溢出 → lse e35 / out e34-NaN / out 的
   q 不变性（饱和）全部由此一条链解释；C01 MLA（SoftmaxFlashV2 + matmul.h 装载）
   finite 通过亦佐证库在正常输入下可用。kw-2 手写 softmax 降级为"可选保险"。

**探针方法论修正（下轮用）**：A''/E 的 dump 在 V 管写后用 MTE3 DataCopy 直读 UB，
缺 V→MTE3 同步（PipeBarrier<PIPE_V> 只同步 V 内部）→ 读到混合旧值。修法：dump 前
SetFlag/WaitFlag<HardEvent::V_MTE3>（或借 outputQue EnQue/DeQue 事件对）。

### P4 kw-4 修复轮（2026-09-02 晚）：三刀三绿两近失，G1/G2 bit 全绿

三构建（kw4a/b/c，每轮单变量），全部按修正版重编纪律：

**kw4a — A 装载修复（根因刀）**：`fia_block_cube_nonquant_gqa_sink.h LoadAToL0` 加
`#if==3510` 分支，改调 matmul.h 310 分支独立函数
`AiInfraInferenceCommonFaBaseMatmul::LoadDataToL0A<T1, ABLayout::MK>`（无 Buffer 框架、
零事件改动；L1 staging 契约由 MLA 在本仓 A5 验证；源基址公式与原实现一致）。验证序列：
q=0 探针 D'/A' 全 0（第一判据过）→ **G1 mqa_c27 bit_exact=True mere=0.0**、G2 同绿。

**kw4b — 跨核同步修复（M>16 竞态刀）**：mode-4 intra-block + wait 落消费队列 + 双 AIV
偏移等待（kw-1 设计重新落码，8 处调用点）。背景：kw4a 后发现 M>16（SetMSplitInfo 双
AIV 分行）非确定性破损（同输入两跑 0.82/0.41）；mode-2 flag 双 setter 时 AIC 只等第
一个 AIV 的 set 即继续。修复后竞态收敛为确定性错（0.29），暴露第三层问题。

**kw4c — M-16 保守分块（多分形刀）**：行带状错定位（M≤16 全对、M>16 带错；带位随
T/vecDealM/mSplitSize 变化无简单分形规律；D' 探针经鉴定为 bf16 视读伪值不可作数值
比对，仅零/非零与 q 敏感度可信）。A5 下 cube 侧 M 维全部按 16 行粒度切分（mm1 的
mL1/mL0 + mm2 的 mL1=mmad m），把 Q staging(ND2NZ)/A 装载/Mmad/fixpipe 全部限制在
M≤16 已验证单分形域。隔离验证 T=16..256/GQA/多循环全 bit-exact。

**六门禁最终（kw4c 产物）**：

| 门禁 | 结果 | 备注 |
|---|---|---|
| G1 mqa_c27 | **bit_exact=True mere=0.0** | |
| G2 mqa_c28 | **bit_exact=True mere=0.0** | |
| G3 C21 | bit=False mere=6.3e-6 max=0.0078 | 从 mere=3.78 收敛至 1 ulp |
| G4 kvlen_4096 | bit=False mere=3.0e-5 max=0.0078 | 1-2 ulp |
| G4 kvlen_32768 | bit=False mere=9.4e-6 max=0.0078 | 1 ulp |
| G5 b2_248 | bit=False mere=1.8e-7 max=0.0039 | 1 ulp（单循环 B=2 累计契约） |
| G6 C01 MLA | finite=True | 同基线口径 |

**余项（bit 未及，数值已对）**：四个近失门禁全部为 1-2 bf16 ulp、mere ≤ 3e-5。
按结构分两类：多 S2 循环 update 路径（4096/32768/C21）与单循环 B=2 累计（b2）。
分析：官方 A5 baseline 的 vec 侧用 ProcessVec1Vf（VF MicroAPI 手写序列，见
`csrc/attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h`），本仓
SoftmaxFlashV2 库调用在单循环 M≤16 形态下与其逐位一致（G1/G2 证明），但多循环
update/累计形态的舍入路径存在 ulp 级差异。**bit 补齐方向 = A5 门控下把
SoftmaxFlashV2Compute 换成 ProcessVec1Vf + UpdateExpSumAndExpMax**（输出含 ND2NZ 变换，
需同步适配 vec1ResGm 写出与 CopyPToL1 读入的 NZ 形态，或取其 T2=float 无 cast 形态；
注意 m≤64 单寄存器约束与 OriginNRange 桶选择）。预计 1-2 轮构建。

### P4 kw-5 bit 补齐轮（2026-09-02 深夜）：条款 5 停止——ulp 基线交付，移植契约静态不可判定

零构建消耗，全部为判别性实验（设备实测）：

**1. 官方基线自洽性验证（门禁参照系无罪）**：官方 B=2 单调用 vs 官方 B=1×2 拼接
= bit 级一致。门禁失败全部是我方与官方的真实差异。

**2. b2_248 定性（值依赖 ulp 边界，非结构性 bug）**：用替代随机数据（Generator(33)
序不同）复刻 b2 形态 → 我方 vs 官方 **bit 级一致**；用门禁精确数据 → mere=1.84e-7
（与门禁 1.8e-7 吻合）。即：特定值落在舍入边界上才现 1 ulp 差。

**3. 余差精确量化（lse 层）**：kvlen_4096 lse 差 1/56、kvlen_32768 差 2/56、C21 形
差 23/960，全部 ≤1 fp32 ulp（1.91e-6 量级）；out 差 0.1%~0.6% 元素、1-2 bf16 ulp。
单循环（c27/c28/b2 替代数据）可 bit 全对 → 病灶在多 S2 循环 softmax update 链的
舍入次序（我方 SoftmaxFlashV2 vs 官方 ProcessVec1Vf MicroAPI 序列），部分边界值
落入不同舍入。

**4. ProcessVec1Vf 移植卡点（条款 5 触发）**：dst/stage1 布局契约静态不可判定——
已确证 vec1Srcstride=(mBaseSize>>1)+1（官方与 FlashMLA 一致，解 bank 冲突 +1 行）、
commonTBuf=512B、m ≤ s1BaseSize/2（每 AIV ≤32 行）、T2=float 需额外
((s1Base/2)+1)×(s2Base/2) x_exp 区。但消费端 DataCopy{blockCount=k/16, blockLen=m,
srcStride=vec1Srcstride-m, dstStride=mBaseSize-m} 的单位语义（元素 vs 32B 块）与
L1 rowSize×16/k-block 的衔接存在多种互斥读法（FlashMLA 实数与两种单位制均可自洽
解释），盲猜错误 = NaN 级回归且危及已绿 G1/G2。**若授权续轮：先做 1 轮
micro-probe 内核（ProcessVec1Vf 注入 m=1 已知模式并 dump dst 区域）钉死布局，
再带置信移植**；或按 ulp 口径降档登记（协调者决定）。

**ulp 基线（降档登记用数值）**：

| 门禁 | bit | mere | max_abs |
|---|---|---|---|
| C21 | False | 6.3e-6 | 0.0078 |
| kvlen_4096 | False | 3.0e-5 | 0.0078 |
| kvlen_32768 | False | 9.4e-6 | 0.0078 |
| b2_248 | False | 1.8e-7 | 0.0039 |

（G1/G2 bit=True；C01 finite=True；数值全部正确，差 ≤1-2 bf16 ulp）

### P4 kw-6 探针轮（2026-09-02 夜）：VF 库接入打通，dst 布局仍未钉死，条款 4 停止

**已固化（提交在案，下轮可直接复用）**：
1. **arch35 VF 库接入基础设施**（3 个坑的解法）：
   - staging：op CMakeLists `set(<op>_depends attention/common CACHE INTERNAL)`（basename 落
     src/common，FlashMLA 同款 depends 机制；本仓 A5 之前没有 op 用它）
   - include：内核源相对路径 `../../common/op_kernel/arch35/vf/...`（op_kernel 是 2 层深，
     FlashMLA 的 arch35 是 3 层深用 `../../../`，不可照抄）
   - 命名空间：vf 头在全局域 `using namespace regbaseutil` 与我方 ConstInfo/RunInfo 歧义
     → 包一层 `namespace FiaVfLib { #include ... }`（工具链头已被前置标准 include 以全局
     域包含，guard 生效，无重复定义）
2. **ProcessVec1Vf 在本仓 FIA 内核内可编译、可运行**：探针（m=2/N=64/GT_0_AND_LTE_64/
   no-update/scale=1）实测 **sum 输出正确**（47.51 = Σexp(0.01(c-63))，逐位符合预期）——
   VF 路径功能可用。
3. 探针的 V→MTE3 dump 同步（SetFlag/WaitFlag<HardEvent::V_MTE3>(5)）实测有效（kw-3 的
   A''/E 探针读数无效问题由此修复；src/slot 回读均正确）。

**新卡点（dst 布局仍不可推断）**：dst tensor 基址区 [0,704B) 被写零、预填魔数残留在
int16 偏移 352 之后、exp 值（~0.87-0.93 一组）疑似出现在 sum slot 区 [1024B,1120B)——
VF 的 strided store（blockStride=(s1Base>>1)|1 编码）实际落点与我读到的所有候选布局
（33 元素制/64 元素制/16×16 分形/528=33×16）均不符。**下一步**（若授权）：把 dump 通道
从 metadata 尾区（1.7KB）换到 GM workspace 尾部（≥16KB），一次 dump tmpBuff1 全量
32KB，直接定位 exp 值与写零区边界，即可钉死布局（预计 1 轮探针 + 1 轮移植）。

ulp 基线不变（kw-5 表）；门禁现状 G1/G2 bit=True、四例 1-2 ulp、C01 finite。

### P4 kw-7 收口轮（2026-09-02 深夜）：硬停止线触发——布局仍未唯一确定，ulp 基线终态

**多窗全量 dump 基础设施（已提交，编译外）**：宿主每次调用前写 `metadata int32[600]=win`
（0..18），内核探针据 `dbgMetaSel.GetValue` 选窗 dump tmpBuff1 的 int16[win×864, +864)
→ metadata int16[1184..2047]；19 次调用拼出全 32KB（`probe_scripts/fia_kw7_sweep.py` +
`/tmp/fia_kw7_full_dump.json` 留档）。

**实测结果（拼装后 16416 int16）**：dst 区仍全零；魔数 0x3F3D 与 exp 序列
（0.5326.. 起的 64 值两种相位）在全 32KB 内**均未找到**；src 模式（0.01×c）出现在位移
位置（拼装 int16≈4096 附近而非窗 0 的 0 偏移）——**dump 拼装一致性本身存疑**（候选解释：
多偶数 AIV last-writer-wins 与窗口选择读的时序互作 / GetWithOffset 窗口偏移与
DumpData 竞争 / VF store 落点超出 tmpBuff1 本体）。按 kw-7 指令硬停止线：
**布局无法唯一确定，立即停止，不追加轮次**。

**终态门禁（=kw-4c 修复后，kw-5/6/7 未再改动功能路径）**：
G1 mqa_c27 / G2 mqa_c28 **bit_exact=True mere=0.0**；G3 C21 mere=6.3e-6、
G4 kvlen_4096 mere=3.0e-5、G4 kvlen_32768 mere=9.4e-6、G5 b2_248 mere=1.8e-7
（全部 1-2 bf16 ulp，数值正确；b2 已证值依赖）；G6 C01 finite=True。
**ulp 降档登记评估交协调者**。

**若未来重启 bit 补齐**（基础设施全就绪）：建议改用"内核内一次性大窗口 dump"——
把 dump 目标从 metadata（1.7KB/窗）换成 attentionOut 尾部未用行（探针测试 shape 可
控制，如 T 加大留尾行），单次 16KB+ 免拼装；或直接读本仓 FlashMLA 已验证的
`stage1OutQue` buffer 布局常量（UB_VEC1_RES = (mBase/2+1)×s2Base×sizeof(INPUT_T)，
mBase=64: 33×128×2B=8.25K 注释自证）反推 VF store 语义。移植约束清单见 kw-5/6 节。

### 探针判读脚本（供协调者复用）

- q=0 快检（lse/log + P 全零判据）：`probe_scripts/fia_round5_check.py`（metadata 通道版）
- 门禁形态隔离（M/kvlen/GQA 扫描）：`probe_scripts/fia_kv_iso.py` 模板（T/N1/N2/kv 参数化）
- C21 分 batch/head 差异模式：`probe_scripts/fia_c21_pat.py`
- 双 q 判读（D'/A'/E 探针）：`probe_scripts/fia_kw3_check.py`
- 重编纪律：本文件 P4 续节（src staging + gen stamps + bin + kernel_meta 全清）

## P3 收尾（2026-09-02）

- 分支提交：`feat/fia-v2-sink-a5` @ `d5c7216ca`（4 文件，+43/-8）
- 交付物：cannbot_handoff.json（a5_port 节）/ GATE_CONSISTENCY.md /
  a5_port_feedback.md（阻塞登记）/ 本笔记 / build_a5_{01..10}.log
- 910_93 回归：本机无设备，静态论证替代（见 handoff json），建议回 A3 环境终验
- 上游回赠候选（构建接线三坑）已列入 handoff json upstream_gift_candidate

### 门禁重建（P2 前置，交接缺口处置）

原工作区 `tests/{gqa_case_builder,case_builder}.py` 未随包携带且原工作区已不存在——
按 pr15336 脚本消费契约**迁移重建**（`tests/` 目录，本包内）：
- `gqa_case_builder.py`：mqa_c27/c28（MQA N2=1 × D192 短/长 kv）、C21（GQA 128/gS2）、
  C22（D192 gS8）、C23（D64 gS4）、C24（B=2 不等长累计契约）；bits_equal 三元组
- `case_builder.py`：C01 MLA（D=512/R=64/kvh=1）
- pr15336 三脚本路径环境变量化（`FIA_GATE_BASE`/`FIA_GATE_TREE`，兑现交接文档"可配"承诺）
- ⚠️ case 参数为重建值（按接口文档 §6 支持矩阵选代表性形态），非原工作区原值；
  rope 分离 GQA 族（128,64,64）与 v≠qk 族（192,128,0）因 pr15336_gate.py 调用模板
  硬编码 `head_dim_v==head_dim_qk`、`rope_head_dim=0` 不覆盖，需专用脚本（遗留项）

## P5 kw-8 MLA 修复轮（2026-09-03，卡 4）：V2 A 装载 + mode-4 同步，全族 3-17× 改善，3 轮预算停

### 环境
卡 2 被 vLLM 占用 → **卡 4**（ASCEND_RT_VISIBLE_DEVICES=4）。所有脚本已换卡。

### 根因行级（MLA 侧与 GQA 侧缺陷对照）

| 缺陷层 | GQA（kw-4 已修） | MLA（本轮修） |
|---|---|---|
| A 装载 | load3d 读错位 L1（LoadAToL0 手搓）→ 换 matmul.h 独立函数版 | MLA 本就走 KPP，但 KPP 310 分支调**独立循环版** LoadDataToL0A（LoadData2DParams 循环）——多 m 分形（singleM>16）确定性破损（与 GQA kw-4 行带状同源）；本轮改走 **V2 泛型**（LoadData2DParamsV2 单发） |
| 跨核同步 | mode-2 双 setter 竞态 → mode-4 intra-block + 消费队列 wait + 双 AIV 偏移 | 同一缺陷原样存在（15 处调用点，含 MLA 特有 C2V1/V1NupdateC2）→ 同款修复 |
| M>16 分形 | M-16 保守分块 | **M-16 分块移植失败**：与 MLA L1 QP4 四缓冲编排（rope 半块拆分 trick 假设 mL1Loops≤2）死锁，已完全回退。V2 装载器本身支持多分形（FlashMLA M=64/128 先例），不需要分块 |
| B 侧装载 | 不动 | V2 切换试验引入 sm3/mask 形态 507015 崩溃（C05），**已回退**留独立版（NK/KN 映射 V2 的 isRightTranspose 推导存疑，见 matmul.h 注释） |

### MLA 全族结果（kw-8c = ccd30efc8，max-rel mere，官方基线，卡 4）

| 组 | 修复前 | 修复后 | 状态 |
|---|---|---|---|
| C04/C08/C11（SigB 典型） | 1.2-2.9 | **7.8e-3（1 ulp）** | ✅ ulp 口径 |
| C18-C20（r96 长KV） | 0.80-0.88 | **7.8e-3（1 ulp）** | ✅ ulp 口径 |
| C05/C06/C07/C10 | 1.2-2.9 | 0.14-0.97 | ⚠️ 中间态未收敛 |
| C12-C17（r12/r24 kv≥128k） | 9.2-75 | 2.3-4.3 | ⚠️ 未收敛（S2-split/FD combine 路径） |

C12-C17 残差结构：同 kv 的 r96 全收敛、r12/r24 不收敛 → 非 kv 长度单一因素，指向
**MLA S2-split/FlashDecode combine 路径**（C04 kv128 无 split 不触发）。候选：
vec 侧 FD combine（lseMaxFd/lseSumFd 融合）在 dav-c310 的算术/同步。

### GQA 回归（kw-8c 构建复核）
G1/G2 bit_exact=True mere=0.0；C21 6.3e-6 / kv4096 3.0e-5 / b2 1.8e-7（与 kw-4 终态
逐位同值）；C01 finite ✓。**GQA ulp 口径不动、不受影响**。

### 性能（本轮顺带，sink vs official 中位）
- MLA 修复后可信：C11 0.146×、C14 0.104×、C17 0.168×（长KV 大幅快于官方）；
  C04 2.26×（小 case launch-bound）
- GQA prefill M-16 分块代价仍在（C23 10.6× 记录，遗留优化项）

### 下一步（若授权 kw-9）
1. C12-C17：FD combine 路径探针（lseMaxFd/lseSumFd GM 中间量 dump，复用 kw-6/7 基建）
2. C05/C06/C07/C10 中间态：mask 路径（sm3）与 V2 A 装载的 rope 拼接交互
3. B 侧 V2 重试：先以 C04 形态 micro-probe 钉死 NK/KN→isRightTranspose 映射

## P6 kw-9 轮（2026-09-03 下午，卡 4）：FD lse 溢出修复 + 尾块重放根因定位，11/16 ulp，硬停止

### 结果总表（kw-9a+9b = fe8d2160e 功能态，max-rel mere，官方对照，卡 4）

| 组 | kw-8c | kw-9 终态 | 状态 |
|---|---|---|---|
| C04/C08/C11 | 1ulp | **1ulp（1ulp=0.0078125；C11 偶发 0.0155 抖动，属残差类）** | ✅ |
| C12-C17（r12/r24 长 KV） | 2.3-4.3 | **1ulp，确定性复现** | ✅ 本轮主战果 |
| C18-C20（r96） | 1ulp | **1ulp** | ✅ 保持 |
| C05/C06/C07/C10 | 0.14-0.97 | 0.14-0.95（未收敛，根因已定位，见下） | ⚠️ |

**11/16 MLA 达 ulp 口径**（kw-8 为 6/16）。GQA 回归：G1/G2 bit=True mere=0.0、
C21 6.3e-6 —— 与 kw-4 终态逐位同值，未回退。

### Bug A（C12-C17，已修）——FD combine lse staging UB 越界

- 逐 case 解码 metadata：C12-C17 的 s2SplitNum=26-28（头少 → FD split 饱和核数），
  其余 case ≤8。FD 假设修正：**FD 在除 C04 外全部 case 激活**（C08/C11/C18-C20 也
  FD 激活且收敛）→ FD 激活不是判别器，split 数才是。
- 定量：C12 sink=3.03×official（全元素，确定性）；误差 ∝ split 数增量（26→2.3,
  27→3.3, 28→4.4）。
- 边界扫描（B1/G8/H16, kv 16k..131k）：splits≤24 全 1ulp，splits≥26 全崩 ——
  边界恰为 fdSum/fdMax/fdLseExp 缓冲容量 6144B = 24×8行×8pad×4B。
- **根因行级**：`fia_block_vec_flashdecode_sink.h InitBuffers` 五个 lse 缓冲
  4K+2K=6144B，容 24 splits；AICPU 校验允许 maxS2SplitNum=aicCoreNum+1=29（A5 28 核）。
  CopyLseIn 的 UB 装载越界 → combine 归一化分母损坏。
- **修法（kw-9a, bf0ff4b5d）**：3510 门控扩到 8K（32 splits）。A3 保持 6144B
  （24 核 → ≤24 splits 恰不越界）。

### Bug B 残差（C05/C06/C07/C10 + C11 抖动，未修，根因链完整）

证据链（全部无构建 host 实验）：
1. lse 全对（≤4e-6）→ softmax/staging/FD 分母无罪；错在**分子（mm2ResGm 累计器偏大）**
2. 脏行 ratio 全 >1（1.05-1.5），缺失质量 w 数据依赖（0.05-0.15）
3. 跨 batch 稳定（同进程 3 跑脏行集逐位同）；bit 级微抖（ulp 噪声叠加）
4. B=1 同 kvlen 全净 → 跨 batch 任务布局相关
5. **尾块窗口**：B8 布局扫尾宽 16..496 → 仅尾宽 ∈[48,128]（=mm2 单 k 块且 >32）脏
6. **杀块实验**：b1 倒数第二个完整 512 块 K 置 -100（P→0）→ mere 塌缩到 1ulp；
   杀更早块/零化下一 batch 池区/零化尾池块 pad → 均无变化
7. 脏 batch 放池末尾（其后全零）→ 净
8. kw-9b（nupdate MTE3→MTE2 核内队列边，79af8275c）无效 → 排除 nupdate 越位假设

**根因（机制级，指令级待探针）**：split 的末 S2 内循环宽 ∈(32,128] 时，mm2 A 侧
L1 tile 仅被 Nd2Nz 刷新 [0..kL1Size)，[kL1Size..256) 残留上一循环（最后一个完整
512 块）的 P；mmad 的 k 扩展消费残留 → 该块贡献被重放（+~1 块权重、恒正、lse 不变）。
杀块 6/7 两实验排除 GM staging 残留（slot 0 残留应为块 L-2，杀块 L-1 才洁）→ 残留
在 L1。

**kw-9c 尝试（375b3de5a，已回滚 fe8d2160e）**：A 侧拷贝定宽 256/行距 s2BaseSize +
V1 尾循环 P staging stride-s2BaseSize + pad 补零。结果净负：C05 0.39→0.071 但
C08/C11 从 1ulp 回退 0.150/0.032、C06 恶化 0.14→0.61 → 布局假设有细节错
（Nd2Nz 256 宽拷贝的 NZ C0 几何或 V1 补零路径），3 轮预算尽，按硬停止纪律回滚。

### 下一步（kw-10 建议，若授权）
1. **先探针后修**：AIC 侧尾 mm2 处 dump 实际消费的 A 值（GM staging + L1 tile），
   选择器走 metadata[601/602]（kw-7 模式）；AIC 标量 GM 读（GetValue）无先例，
   需先验证原语
2. 候选修法方向：尾 k 块的 L1 tile 整块刷新（几何细节需探针钉死）；或参考
   FlashMLA arch35 的 UB 内累计（无 GM 原子、无此缺陷类）
3. C11 抖动（0.0078↔0.0155）与 C05 组同源，修尾块后应一并消除

### kw-9 期间 incidental 发现
- sink 算子 V≠K（独立 value tensor）寻址疑坏：指示 V 实验中 sink 输出恒 0（读错池区）
  而官方输出正常用 K 当 V —— 全部 16 case 均 V=K，不影响战役真值，登记遗留
- /tmp 会被 torch_npu 加载器的 .fed*.so 残留撑满（327 个 1.4G）→ makeself 打包失败
  "failed to create temporary archive"；解法：TMPDIR 重定向到大分区（本轮
  tmp_build/），勿再犯

### kw-9 提交链
- bf0ff4b5d kw-9a：FD combine lse staging UB 溢出（C12-C17 修复）
- 79af8275c kw-9b：AMLA nupdate→V2 读 MTE3/MTE2 核内队列边（防御性，对该残差无效，
  保留：语义正确且配平，无回归）
- 375b3de5a kw-9c：尾块重放修法尝试（净负）
- fe8d2160e revert kw-9c（最终交付态 = kw-9a+9b）

## P6.5 kw-10 轮（2026-09-03 晚，卡 4）：尾宽膨胀修复 C05-C07/C11，15/16 ulp 终态

### 终态总表（kw-9a+9b+kw-10+安全钳位 = cb79e16d9 功能态，max-rel，2 跑）

| case | mere | 状态 | | case | mere | 状态 |
|---|---|---|---|---|---|---|
| C04 | 0.0078 | ✅ | | C11 | 0.0078（×5 稳） | ✅ |
| C05 | 0.0078 | ✅ | | C12-C20 | 0.0078 ×9 | ✅ |
| C06 | 0.0078 | ✅ | | C10 | 0.95 | ⚠️ 残差（见下） |
| C07/C08 | 0.0078 | ✅ | | | | |

**15/16 MLA 达 ulp 口径且确定性复现**。GQA 回归：G1/G2 bit=True mere=0.0、
C21 6.3e-6 —— 与 kw-4 终态逐位同值。

### 修法一（kw-10, 0c8aa4290）：S2 尾循环宽度膨胀到完整 512

- 路线 a 前置核查（代码证据）：MatmulKPP 内 `mmad k = kSplitSize`（实际宽）、
  L1→L0A 装载用 `AlignUp(k,16)` —— 对 16 倍数尾宽（64/96/128）装载==mmad==实际宽，
  源码层无对齐差 → 路线 a 前提不成立，改为「把 mmad 将消费的窗口整体刷成当前块」
  的最小变体：**CalcParams 单一真值点把部分尾循环（0<w<512, 有 mask）的处理宽
  膨胀到 s2BaseSize**。pad 列被 sm3 mask 置 -inf（P=0、lse/分母不变），其 K/V 经
  bt padding 读 0 号池块。完整循环是一切已测布局从未脏的普适类；sm0（无 mask）
  不膨胀（现网 sm0 case 尾宽均 0）。
- 验证：C05/C06/C07 → 1 ulp 确定性；C11 五跑全稳（kw-9 时的 0.0155 抖动消除）；
  C04/C08/C12/C18 无回归。

### 安全钳位（kw-10 附带，提交 cb79e16d9）

尾宽膨胀的 pad 列 K/V 装载会请求超出本 batch 槽位的块号；当本 batch 即最宽
batch 时**越过 block table 张量宽度** → 间歇 507015（首次构建侥幸 3 过、重建即崩，
分配器相关）。修法：`BlockTableParser::GetBlockIdx`（memory_copy.h，唯一 bt 取块点）
钳位到 maxblockNumPerBatch-1；非膨胀路径块号恒小于宽度（行为不变）；pad 列数据被
mask，值无意义仅需防越界。修复后 C05 累计 5 跑全过。

### 修法二尝试（kw-10b/10b2，f487f52fa/f9fb7b9df，已回滚 42cdc495a）：C10 小 M 垫宽

- C10 判别（host 实验）：G×H 扫描实锤 **M≤32 专属**（G=1 mere 1.85 / G=2 ~1.0 /
  G=4/8 全净）；m 边界扫描 m=16/32/40/48/64 脏、m=56(C04)/128 净；lse 全对（分子错）；
  误差按 **D 维 n-chunk 单调递增**（nL1=0 净 → nL1=3 最脏 59%，行内元素要么精确
  1.0000 要么 ~1.11 二值分布）。
- kw-10b（mm2 处理 m 垫到 64）与 kw-10b2（垫到 128 规范几何）均无效（C10 原值
  0.97、chunk 模式不变）→ **mm2 立方链对小 M 排除**。两提交已回滚（无效实验不留树）。
- C10 残差定性（kw-11 建议）：缺陷在 V1/V2 AIV 侧小 m 路径（vecDealM=16 的对半）
  或 nupdate 除数链 —— D-chunk 单调 + lse 净 + mm2 链排除。建议探针：dump V1 的
  aMlaSumUb/tmpSumUb（V2 除数）与 V2 读值，metadata[601/602] 选择器模式。

### 性能（收敛后，sink/官方中位，卡 4）

| case | ratio | | case | ratio |
|---|---|---|---|---|
| C05 | **0.720** | | C11 | **0.139** |
| C06 | **0.551** | | C12-C20 | 0.107-0.592（见 P6 表） |
| C07 | **0.339** | | C10 | 0.691（数值未收敛，仅参考） |

全部收敛 case 快于官方；尾宽膨胀代价可忽略（C05 0.817→0.720 反而更快，官方侧
同 case 波动）。

### kw-10 提交链（feat/fia-v2-sink-a5 终态）
- 0c8aa4290 kw-10：S2 尾循环宽度膨胀（C05/C06/C07/C11 修复主体）
- f487f52fa/f9fb7b9df kw-10b/b2：mm2 小 M 垫宽（无效，已回滚）
- 42cdc495a revert kw-10b/b2
- cb79e16d9 安全钳位（bt 越界修复）—— 最终交付态

### A3 隔离论证（kw-10 全部改动）

尾宽膨胀（CalcParams）、bt 钳位（memory_copy.h）全 `#if __NPU_ARCH__==3510` 门控；
A3 编译路径逐字节不变；契约零变更；GQA 专属文件零触碰（G1/G2/C21 终验逐位同值）。

### 性能（C12-C20 全部 1ulp 后复测，可信）

| case | sink(ms) | official(ms) | ratio |
|---|---|---|---|
| C12 | 0.373 | 1.928 | **0.194** |
| C13 | 0.518 | 3.832 | **0.135** |
| C14 | 0.801 | 7.514 | **0.107** |
| C15 | 0.515 | 1.960 | **0.263** |
| C16 | 0.737 | 3.874 | **0.190** |
| C17 | 1.222 | 7.639 | **0.160** |
| C18 | 1.192 | 2.014 | **0.592** |
| C19 | 2.154 | 3.898 | **0.553** |
| C20 | 4.066 | 7.674 | **0.530** |

收敛的 9 个长 KV case 全部快于官方（3.5-9.4×）；C05 组性能待修复后复测。

### A3 隔离论证（kw-9 全部改动）

kw-9a/9b/9c(已回滚) 全部位 `#if __NPU_ARCH__==3510` 门控；A3 (c220) 编译路径
逐字节不变（kw-9a 的 6144B 保留、kw-9b 的 MTE3_MTE2 边与 kw-9c 布局改动均 3510-only，
后者已整体回滚）。契约零变更（无 schema/tiling/binding 改动）。
