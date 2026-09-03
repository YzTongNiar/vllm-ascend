# FIA v2 Sink / Metadata 算子接口与实现说明

> 版本：round_5 结档态（对应交付包 `handoff/d192_fix2_delivery/`，2026-08-28）
> 适用二进制：d192_fix2_delivery（OPP + wheel 成对）；CANN 9.1.0 构建链路（9.2 兼容）
> 权威包络登记：`results/round_3/r3_w1_gqa_boundary.json`（update_round_4 节）；本文与其不一致时以 boundary json 为准

---

## 1. 架构概览

从 omni-ops `ai_infra_fused_infer_attention_sink(+_metadata)` 移植的**双算子形态**（OKF 策略 X47：AICPU tiling 下沉）：

```
┌────────────────────────────────────────┐ meta ┌─────────────────────────────────────┐
│ FusedInferAttentionScoreV2SinkMetadata  │ ───► │ FusedInferAttentionScoreV2Sink      │
│ (AICPU, DNN_VM_AICPU)                   │ GM   │ (AICore, MIX_AIC_1_2)               │
│ device 读 seq_len → SplitCore 同构分核  │      │ 读 GM 元数据驱动分核执行注意力      │
│ 写 uint32[1024] 元数据                  │      │ (host tiling 仅 shape-only)         │
└────────────────────────────────────────┘      └─────────────────────────────────────┘
```

- **设计目的**：seq_len 全程留在 device，D2H/H2D/aten::item 归零（profiler 实证）；主算子 host tiling 与数据值解耦（同 shape 不同值 → tiling 输出逐字节相同）。
- **ACLNN 名称**：`aclnnFusedInferAttentionScoreV2SinkMetadataV3`（前置）→ `aclnnFusedInferAttentionScoreV2SinkV3`（主，V3 版本号与官方 FIA 对齐）。
- **torch 入口**：`torch.ops.custom._npu_fused_infer_attention_score_v2_sink_metadata` / `torch.ops.custom.npu_fused_infer_attention_score_v2_sink`。
- **与官方 `npu_fused_infer_attention_score_v2` 的关系**：语义对齐（host list 传参基线改为 device tensor + 两段式），数学主体一致；门禁以与官方逐 bit 对照为标准。

## 2. 部署

```bash
# 1) torch 扩展（wheel 内含 python 入口与 aclnn 绑定）
pip3 install fia_v2_sink_ops-1.0-cp311-cp311-linux_aarch64.whl

# 2) 自定义算子库（双条目缺一不可：条目1 供 runtime 注册/加载 AICPU so，
#    条目2 供 torch 扩展解析 aclnn 符号 libcust_opapi.so）
export ASCEND_CUSTOM_OPP_PATH=<包>/packages:<包>/packages/vendors/omni_custom_transformer

# 3) CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh        # 9.1 链路（本仓构建链）
# 或 /home/t00886357/ascend-9.2.0/cann/set_env.sh          # 9.2（9.1 构建包免重建兼容）
```

## 3. 主算子接口：`npu_fused_infer_attention_score_v2_sink`

### 3.1 输入（30 项，★ = Sink 服务形态必填/常用）

| # | 输入 | 形状/类型 | 说明 |
|---|---|---|---|
| 0 | ★ query | [T, N1, D]（TND）或 [N1, T, D]（NTD_TND） | T=Σqlen；bf16/fp16 |
| 1 | ★ key | PA 池 4 维 [pool, N2, BS, D]（BnNBsD）或 3 维 [pool, BS, N2×D]（BnBsH/BBH）；非 PA 连续 KV 亦支持 | bf16/fp16 |
| 2 | ★ value | 同 key（独立张量，GQA 非 MLA-absorbed） | |
| 3 | pse_shift | 可选 | 未启用 |
| 4 | atten_mask | 可选 | 未启用 |
| 5 | ★ actual_seq_lengths | **device INT64 [B]，TND 语义 = 累计值** [7,14,…] | ⚠️ 关键契约，见 §7.1 |
| 6 | ★ actual_seq_lengths_kv | device INT64 [B]，逐请求值 [20,45,…] | |
| 7–13 | dequant_scale1 / quant_scale1 / dequant_scale2 / quant_scale2 / quant_offset2 / antiquant_scale / antiquant_offset | 量化链路 | GQA 非 quant 不传 |
| 14 | ★ block_table | INT32 [B, bt_w]，页号（bt_w = 单请求最大页数） | PA 必填 |
| 15/16 | query_padding_size / kv_padding_size | 可选 | 未启用 |
| 17–20 | key/value_antiquant_scale/offset | 量化链路 | 不传 |
| 21/22 | ★ query_rope / key_rope | MLA 分离 rope | GQA 传 **None**（rope_head_dim=0 自动落 GQA 分支）；MLA 必传 |
| 23/24 | key_rope_antiquant_scale / dequant_scale_query | 量化链路 | 不传 |
| 25 | ★ **metadata** | uint32[1024]（前置算子输出） | **必填——缺失直接 aclnn 失败（两段式契约）** |
| 26–29 | learnable_sink / key_sink / key_rope_sink / value_sink | sink token 张量 | 当前门禁族未启用 |

### 3.2 输出（4 项）

| 输出 | 形状 | 说明 |
|---|---|---|
| attention_out | [T, N1, D]（TND；与 query 输入布局对应） | 与 q 同 dtype |
| softmax_lse | [N1, T] fp32（`return_softmax_lse=True` 时） | |
| softmax_max / softmax_sum | MLA 吸收方案专用（rope_split + vHeadDim=512） | GQA 不用 |

### 3.3 属性（19 项）

`num_heads`(N1)、`scale`(=1/√D)、`pre_tokens`、`next_tokens`、`input_layout`("TND"/"NTD_TND"/"BNSD"/"BSH"…)、`num_key_value_heads`(N2)、`sparse_mode`(已验证 0)、`inner_precise`、`block_size`(128)、`antiquant_mode` 系列、`softmax_lse_flag`、`pse_type`、`sink_number`、`batch_invariant`、`softmax_lse_max_sum`、`out_dtype`

## 4. 前置算子接口：`_npu_fused_infer_attention_score_v2_sink_metadata`

| 项 | 内容 |
|---|---|
| 输入（标量 6） | `num_heads_q`(N1)、`num_heads_kv`(N2)、`head_dim_qk`(D)、`head_dim_v`(vD) |
| 输入（tensor 2） | `actual_seq_lengths`（device INT64 [B] 累计）、`actual_seq_lengths_kv`（device INT64 [B]） |
| 输出 | `metaOut` uint32[1024]（布局：aic[36][10] 任务表 + aiv[72][3] + base[10]，见 `op_kernel/fused_infer_attention_score_v2_sink_metadata.h` 索引定义） |
| 属性（13） | `batch_size`、`sparse_mode`、`pre/next_tokens`、`input_layout`、**`input_layout_kv`**、`sink_num`/`k_sink_num`、`rope_head_dim`、`block_size`、**`aic_core_num`/`aiv_core_num`**、`batch_invariant` |

**契约与陷阱**：
- **F3**：`input_layout_kv` **勿传 "TND_NTD"/"NTD_TND"**（AICPU 字面量解析缺口，errcode 0x2a）；传 "PA"/"TND" 等价形态。
- **aic/aiv 核数必须由调用方从 `torch.npu.get_device_properties().aic_core_num` 等取**，勿用 schema 默认值（wrapper `metadata()` 已自动处理）。
- `batch_size=0` 表示从 seq len 张量长度推导。

### 4.1 metadata 输出内容、作用与计算流程

#### 4.1.1 metaOut 布局（uint32[1024]，GM 上的“每核任务书”）

| 段 | 尺寸 | 内容 |
|---|---|---|
| `aicMetadata[36][10]` | 360 | **每个 AIC 核的任务区间表**。字段序：`CORE_ENABLE / BN2_END / GS1_END / S2_END / BN2_IDX_OF_FD_HEAD / GS1_IDX_OF_FD_HEAD / S2_SPLIT_NUM_OF_FD_HEAD / S2_SPLIT_START_IDX_OF_CORE / GS1_SPLIT_NUM_OF_FD_HEAD / GS1_LAST_PART_SIZE_OF_FD_HEAD` |
| `aivMetadata[72][3]` | 216 | **AIV（向量核）侧 FD 表**：`CORE_ENABLE / GS1_IDX_END_OF_FD_HEAD / GS1_IDX_END_OF_FD_HEAD_SPLIT` |
| `baseMetadata[10]` | 10 | 全局参数：`M_BASE_SIZE / S_INNER_SIZE / M_FD_BASE_SIZE / NUM_OF_FD / USED_CORE_NUM / USED_VEC_NUM_OF_FD / S1_SIZE / S2_SIZE / ACTUAL_LEN_Q_DIM / ACTUAL_LEN_KV_DIM` |

寻址宏（主算子侧共享头 `op_kernel/fused_infer_attention_score_v2_sink_metadata.h`）：
`GetAICMetaAbsIndex(core, idx) = core×10+idx`、AIV 段偏移 360、base 段偏移 576。

#### 4.1.2 作用：把“分核决策”从 host 搬到 device

host tiling 若要分核必须知道 seq_len **数值** → 数值在 device 上动态变化 → 陷入 D2H/图重绑。
本算子的解法：host tiling 只做 shape 级准备；**AICPU 前置算子在 device 上读 seq_len 张量、
跑与 host 同构的负载均衡算法，把“哪个核算哪些 (b, n2, gS1, s2) 任务”写成本表**；
主算子 AICore 启动后每核从 GM 读自己的任务区间直接开算。seq_len 变化 = 任务表变化，
全程无 D2H、无图重绑（X47 双算子下沉的核心价值）。

#### 4.1.3 为什么需要 actual_seq_lengths（q，累计）与 actual_seq_lengths_kv（kv，逐请求）

这两个张量是 AICPU **唯一的数据值来源**，决定三件事：

| 用途 | 取值方式 |
|---|---|
| `bSize`（批数） | 张量长度（`batch_size=0` 时自动派生；同时写 `ACTUAL_LEN_Q_DIM/KV_DIM` 传给主算子的 lens 解析器） |
| 每请求 s1Size | **TND 累计差分**：`s1[b] = cum[b] − cum[b−1]`（b=0 直读 `cum[0]`） |
| 每请求 s2Size | **逐请求直读**：PA 模式下 kv len 是逐请求值 `s2[b] = kv[b]` |

s1/s2 驱动任务块计数（`s1GBaseNum=⌈s1×g/mBase⌉`、`s2BaseNum=⌈s2/sInner⌉`）与负载代价
（`CalcBatchCost`），是分核计划的全部输入。

⚠️ **契约（实测教训）**：若把逐请求 q-len 误当累计传入，`s1[b1]=cum[b1]−cum[b0]` 会算出
0/负数 → 对应任务被 SKIP → **非首请求输出全零，无任何报错**。正确示例：两请求各 7 token
→ `actual_seq_qlen=[7,14]`。

#### 4.1.4 AICPU 内部计算流程（`*_metadata_aicpu.cpp`）

```
① 属性解析      num_heads(q/kv)、head_dim_qk/v（供 rope/MLA 判定）、layout、sparse、
                block_size、aic/aiv 核数
② CalcInnerSizeGqa
                TND: sInnerSize=512（clamp 到 s2）、mBaseSize=512（TND 恒定，与 D 无关）
③ CalcSplitInfo 每 batch: s1GBaseNum=⌈s1×g/mBase⌉、s2BaseNum=⌈s2/sInner⌉
                → totalBlockNum = Σ_b s1GBaseNum[b]×s2BaseNum[b]×n2
④ CalcCostInfo  每 batch 负载代价（s1G 分块 × s2 的块代价累计，CalcBatchCost）
⑤ 方案搜索      i ∈ [⌈√totalBlockNum⌉, min(AIC核数, totalBlockNum)]，每个 i 跑
                CalcSplitPlan（贪心：AssignByBatch → AssignByRow →(FD) AssignByBlock/
                ForceAssign，每核代价上限=剩余代价/剩余核数），保留 maxCost 最小的方案
⑥ SplitFD       若存在 FD 头（任务数 < 核数，s2 跨核切分做 decode 负载均衡），
                生成 FD 头表（BN2_IDX_OF_FD_HEAD / S2_SPLIT_NUM / GS1_LAST_PART_SIZE …）
⑦ GenMetaData   写 36×10 任务表 + 72×3 AIV 表 + base 10 字段 → GM uint32[1024]
```

#### 4.1.5 主算子（AICore）如何消费

- 每 AIC 核 i 读自己的 `bN2End[i−1]…bN2End[i]` 作为任务区间，游标按 (bIdx=bn2/kvHeadNum,
  n2, gS1, s2) 推进（`GetTaskDealMode`/`CalcParams`），跨 batch 时经 `UpdateKey/UpdateValue`
  切换 KV 基址；
- base 段的 `mBaseSize/sInnerSize` 直接决定主 kernel 的内循环尺寸；AIV 侧用 aivMetadata
  的 FD 表做 decode 合并（`accumOut/lse`）；
- 实测（D192 g=1 B=2，累计 q=[7,14]，kv=[20,20]）：`MBase=512 SInner=20 NumFd=0
  UsedCore=16`，`bn2End=[1,2,…,16]`（16 任务/16 核，每核 1 任务）。

#### 4.1.6 注意事项

1. metadata 是**每 forward 一次**的临时任务书（uint32[1024]，4KB GM），非持久配置；
   AICPU 为异步执行——调用方若复用输入 buffer，需保证不被后续步骤覆盖（PR wrapper 用
   per-call clone，见 §8 注记）。
2. `NUM_OF_FD>0`（FD 分核）时任务表含 FD 头字段，主算子走 flash-decode 合并路径；
   `NUM_OF_FD=0` 时为常规分核。
3. 空 batch（s1 或 s2 为 0）不产生任务块；`isKvSeqAllZero` 时退化为单核空跑。
4. q-len 累计契约见 §7.1；kv-len 为逐请求值。

### 4.2 metaOut 的存放位置与形态（device GM、定长张量、非 list）

**位置：全程在 device（NPU GM 显存），数值不回 host。**

| 环节 | 位置 | 依据 |
|---|---|---|
| 计算 | **device 上的 AICPU**（NPU 芯片内部 CPU 子系统，非 host CPU） | 算子注册 `engine=DNN_VM_AICPU`；异常报 "AICPU kernel execution failed" |
| 输入（seq 张量） | device GM | 入参为 device INT64 张量 |
| **输出 metaOut** | **device GM**（一个 NPU tensor） | wrapper 直接返回算子结果；解码需先 `.cpu()` |
| 消费 | 主算子 AICore 经 `metadataGm.GetValue()` **直接从 GM 读** | 两段式契约：`meta_data=meta` 传 device 指针 |

host 全程只做两件事：分配输出张量的“壳”、把 tensor 指针传给主算子——**数值永远不上 host**。
这是该设计的目的（零 D2H），也是图模式能 DYNAMIC_OK 的前提（重放前在 device 上原地刷新这块 buffer）。

**形态：不是 list，是“三段结构体扁平化”后的定长一维张量。**

```python
meta.shape == (1024,)    # dtype uint32/int32，共 4KB，device='npu'
```

内容 = 三段拼接在一个 flat buffer（布局表见 §4.1.1）：

```
[0   ..359]  aicMetadata[36][10]   每 AIC 核任务区间表
[360 ..575]  aivMetadata[72][3]    AIV 侧 FD 合并表
[576 ..585]  baseMetadata[10]      全局参数
[586 ..1023] 保留
```

主算子不按“list 元素”读它，而是按**结构体字段语义**用偏移宏寻址：
`GetAICMetaAbsIndex(core, idx) = core×10+idx` 等（共享头
`op_kernel/fused_infer_attention_score_v2_sink_metadata.h`，含
`static_assert(1024×4B ≥ sizeof(FiaSinkMetaData))`——1024 为定长契约）。
调试时 `.cpu().tolist()` 得到 1024 个整数的 Python list，那只是查看方式；
字段级读取示例：`m[c*10+1]` = 第 c 核的 BN2_END。

**生命周期**：

- eager：每次 forward 由 metadata 算子新建一个输出张量（wrapper 支持外部传 `meta_data`/`output_buffer` 复用）
- 图模式（PR 设计）：写进**持久静态 buffer**（per-num-reqs），重放前原地刷新内容、指针不变——绕开 F1 的关键
- 竞态注记：该 buffer 被 AICPU 异步读，跨流/复用覆盖会污染分核计划（跨流竞态已设备实测复现）——clone/双缓冲防的就是这个（§8 注记、round4_patches）

## 5. 调用方法（三选一）

```python
# ① 推荐：entry() 两段式封装（python/fia_v2_sink_entry.py）
from fia_v2_sink_entry import entry
out, lse = entry(q, kpool, vpool,
    query_rope=None, key_rope=None,             # None → rope_head_dim=0 → GQA 分支
    num_query_heads=N1, num_key_value_heads=N2, # N2=None 时从 4 维池 shape[1] 自动派生
    input_layout="TND", input_layout_kv="PA",   # KV 连续布局按实际传（如 "TND"）
    sparse_mode=0, softmax_scale=1.0/math.sqrt(D),
    block_table=bt, block_size=128,
    actual_seq_qlen=qlen_dev_cum,               # device INT64 累计值 [7,14,...]（§7.1）
    actual_seq_kvlen=kvlen_dev,                 # device INT64 逐请求值 [20,45,...]
    return_softmax_lse=True)                    # → (out, lse)

# ② 显式两段（图模式复用 meta / 冒烟）：metadata() → torch.ops
from fia_v2_sink_entry import metadata
meta = metadata(N1, N2, D, vD, qlen_dev, kvlen_dev,
                input_layout="TND", input_layout_kv="PA", rope_head_dim=0,
                block_size=128, aic_core_num=aic, aiv_core_num=aiv)
out, lse = torch.ops.custom.npu_fused_infer_attention_score_v2_sink(
    q, k, v, query_rope=None, key_rope=None,
    actual_seq_qlen=qlen_dev, actual_seq_kvlen=kvlen_dev,
    block_table=bt, meta_data=meta,             # 必带
    num_query_heads=N1, num_key_value_heads=N2,
    input_layout="TND", sparse_mode=0, softmax_scale=scale, block_size=128,
    return_softmax_lse=True)

# ③ SinkGraph：aclgraph 捕获/重放封装 —— ⚠️ F1 FROZEN（失败模式已冻结）
#    平台限制：图重放不重执行 AICPU 定制节点 + 主算子捕获形态烘焙分核方案，
#    动态 seq 更新无法经图重放生效（9.1/9.2、MLA/GQA 同限）。
#    生产链路请使用 eager（①/②）。
```

## 6. 支持矩阵（当前二进制，round_5 结档）

| 维度 | 支持 | 验证状态 |
|---|---|---|
| 数据类型 | fp16 / bf16（非量化；q=kv=out 同型；lse fp32） | C01–C26 门禁 bit 绿 |
| GQA head 维度 (qk, v, rope) | (128,128,0)、(64,64,0)、(128,64,64)、**(192,128,0)、(192,192,0)**（round_4 新增 104 GENERAL 族） | D192 全套门禁 bit 绿 |
| MLA | qk=v=512 + rope=64（n2=1，rope 分离传输） | C01–C20 bit 绿 |
| q 布局 | TND、NTD_TND、BNSD、BSH/BSND（tiling key 全覆盖） | TND/NTD_TND bit 绿 |
| KV 形态 | PA 池 4 维 BnNBsD / 3 维 BnBsH（BBH）；非 PA 连续 KV | 两种 PA 池 bit 绿 |
| gSize（=N1/N2） | 运行时量，不进 tiling key：1–8+ 均可（MHA g=1 到 gS=8） | gS{1,2,4,6,8} bit 绿 |
| batch | **≥1 全支持**：B=2 等长/不等长、B=4 混合变长、B=8 已 bit 验证（对照官方逐请求 B=1 concat） | round_5 |
| kvlen | B=1 至 **32K**（bit 级）；B≥2 ≤248 bit 级；B≥2 ≥4K 见 §8 注记 | round_5 |
| sparse_mode | 0（门禁覆盖）；1–4 的 key 存在未在门禁族 | sm=0 |

**边界（不支持）**：
- TND_NTD 组合布局 GQA（无对应 tiling key）；
- 量化输入（GQA 族仅非量化；量化链路输入存在但未在门禁族）；
- vHeadDim 与上述组合不一致的形态（host `CheckGqaDSupport` 路由即拒）。

## 7. 输入契约与注意事项（排查优先级从高到低）

1. **TND 累计 q-len 契约**：`actual_seq_qlen` 必须传**累计值** `[qlen0, qlen0+qlen1, …]`。传逐请求值会导致解析器差分出 0/负数 → 任务被 SKIP → 非首请求输出**全零**（无任何报错）。排查 B≥2 异常第一步：核对调用方输入构造（正确示例见 `tests/gqa_case_builder.py` 的 `qlen_cum`）。
2. **两段式契约**：主算子必须带 `meta_data`（无 meta 直接 aclnn 失败）；aic/aiv 核数从 device properties 取。
3. **F3**：metadata 的 `input_layout_kv` 勿传组合布局字面量（"TND_NTD" 等），传 "PA"/"TND"。
4. **F1（eager-only）**：aclgraph 重放不重执行 AICPU 定制节点 + 主算子烘焙分核 → 动态 seq 更新在图模式下无效。生产链路 eager。
5. **F2（构建纪律）**：网络盘/NFS 上构建有竞态，必须 `cp -a` 到本地盘构建。
6. **部署双条目**：`ASCEND_CUSTOM_OPP_PATH` 两条缺一不可。

## 8. 行为注记（非缺陷）

| 场景 | 现象 | 性质 |
|---|---|---|
| D192 × B≥2 × kvlen≥~4K | sink 的 FD 分块负载均衡与官方 B=1 不同 → fp32 归约序差，mere ~2e-05 | ≪ bf16 分辨率 3.9e-3 与 P2 门限 7.8e-3；kvlen≤数百为 bit 级一致 |
| launch-bound 短负载 | 每 call 固定成本 ≈ AICPU 130–317µs + 双 aclnn 串行 150–190µs → 净亏 | 主 kernel ≳500µs 的 kernel-bound 负载净平/微赚（X47 成本判据） |
| golden（fp32 教科书）对照 | MERE_golden ~0.17（D192 bf16） | P2 参考口径，官方算子同偏；不作门禁 |

## 9. 关键路径

| 项 | 路径 |
|---|---|
| python 入口 | `python/fia_v2_sink_entry.py`（entry / metadata / SinkGraph） |
| 算子源码 | `op_src/src/ops-transformer/attention/fused_infer_attention_score_v2_sink{,_metadata}/` |
| 公共库（offset/memcopy/tiling 模板） | `op_src/src/ops-transformer/attention/common/` |
| 门禁与探针 | `tests/`（gqa_case_builder / check_gqa / run_gate_w1）+ `results/round_5_fix2/`（gate_kvlen / gate_b2_cum / gate_bN_cum） |
| 包络权威登记 | `results/round_3/r3_w1_gqa_boundary.json`（update_round_4 节） |
| 交付物 | `handoff/d192_fix2_delivery/`（round=2, status=done）；回滚基线 `results/round_4_d192/rollback/` |
| OKF 知识 | `skills/evolution-knowledge/references/okf/`（X47 / 移植 runbook / F1 / F2 / F3） |

---

*维护约定：本文档随交付轮次更新；接口契约变化（schema/entry 签名）必须同步修订 §3/§4 并在 handoff json 中显式声明。*
