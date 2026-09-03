# 门禁一致性表（910_93 回归 × ascend950 新验）— kw-4 修复后全量数据（2026-09-02）

口径：bit 级（vs 官方 `torch_npu.npu_fused_infer_attention_score_v2` 同机对照；
G5 为 vs 官方逐请求 B=1 concat；G6/MLA 族为可运行+有限值）。
A5 列 = kw-4 修复版（HEAD `22faaa0d1` 功能态，装机 kw4c 产物），950PR 卡 2。

## 核心六门禁

| 门禁 | 形态 | 910_93 基线 | ascend950（kw-4 修复后） | 一致性 |
|---|---|---|---|---|
| G1 mqa_c27 | D192 MHA g1 B1 kv20 | bit=True | **bit=True mere=0.0** | ✅ |
| G2 mqa_c28 | D128 MHA g1 B1 kv20 | bit=True | **bit=True mere=0.0** | ✅ |
| G3 C21 | GQA gS2 D128 B8 变长 | bit=True | bit=F mere=6.3e-6 max=1 ulp | ⚠️ ulp |
| G4 kvlen_4096 | D192 g1 B1 | bit=True | bit=F mere=3.0e-5 max=1 ulp | ⚠️ ulp |
| G4 kvlen_32768 | D192 g1 B1 | bit=True | bit=F mere=9.4e-6 max=1 ulp | ⚠️ ulp |
| G5 b2_248 | D192 B2 kv248 | bit=True | bit=F mere=1.8e-7 max=1 ulp | ⚠️ ulp（值依赖，替代数据 bit 全对） |
| G6 C01 MLA | D512+R64 B8 mask sm3 | 可运行+有限 | finite=True out=(64,16,512) | ✅ |

## 扩面（kw-4 修复版新测）

| 项 | 结果 |
|---|---|
| C22–C26（GQA 族） | 全 bit=F，mere 2.0e-6~3.4e-5，max 1-2 ulp（同性质舍入序差，无新形态错误） |
| MLA C02/C03/C09 抽测 | 全 finite=True、shape 正确（SigB dense / SigA min / varlen） |
| q=0 受控快检 | **56/56 过**（lse 全=log(kvlen)，out≈V 均值 1 ulp 内） |
| 隔离扫描（kw-4） | T=16..256 / MHA / GQA(gS2) / 多 S2 循环 bit-exact |

## ulp 差性质（kw-5 判别定性）

- 单循环形态（c27/c28/b2 替代数据）**bit 全对** → 非结构性 bug
- lse 层差 ≤1 个 fp32 ulp（kv4096 差 1/56、kv32768 差 2/56、C21 差 23/960 元素）
- 病灶 = SoftmaxFlashV2 与官方 ProcessVec1Vf 的**多 S2 循环 update 舍入次序差**
- 与接口文档 §8 A3 已知注记（归约序差 mere~2e-5）同性质、量级更小

## 性能（sink 两段式 vs 官方，wall-clock 中位，20 轮，卡 2）

| 用例 | sink (ms) | official (ms) | ratio | 判读（X47 成本判据） |
|---|---|---|---|---|
| S1 r96 gS8 kv256K | 4.076 | 4.769 | **0.855** | kernel-bound **净赚 14.5%** |
| S2 r96 gS8 kv512K | 7.968 | 9.550 | **0.834** | kernel-bound **净赚 16.6%** |
| kvlen_4096 | 0.159 | 0.068 | 2.343 | launch-bound 净亏（固定成本主导） |
| g1_short (kv20) | 0.161 | 0.057 | 2.826 | launch-bound 净亏（同 A3 行为注记 0.12-0.7×） |

**A5 kernel-bound 提升幅度（14-17%）显著优于 A3 基线（~3%，X47 卡 C19/C20 记录）**。

## 910_93 回归

本机无 910_93 设备，未实测。静态论证：kw-1..7 全部改动位于 `#if __NPU_ARCH__==3510` /
`FIA_A5_DEBUG_DUMP`（编译外）/ A5 条件 depends staging 内，A3 路径逐字节不变。
建议回 A3 环境跑全量门禁终验（自 kw-1 起 A3 未编译验证）。

## bit 补齐遗留（未竟项）

ProcessVec1Vf dst strided store 布局两次独立探针（1.7KB 窗 / 32KB 多窗拼装）均未唯一钉死
（多窗拼装一致性存疑）。重启路径留档于 MIGRATION_NOTES.md P4 kw-7 节：
attentionOutput 尾行单次大窗 dump 免拼装 / 从 FlashMLA `UB_VEC1_RES` 自证注释反推。

---

## 附：全族扩测（2026-09-03，卡 4 干净环境；此前卡 2 被 vLLM 占用数据作废）

### 环境勘误
- 卡 2 已被 vLLM 占用（98GB）——09-03 晨卡 2 sweep 全部作废（挂死/大 mere 均系并发污染）
- 晨报"GQA_C21-C26 挂死"实为 sweep 脚本 bug（gqa_parse 缺 resolve，TypeError 被误标），已修
- 数据文件：perf/family_results_card4.txt（22 case，逐 case 子进程隔离）

### GQA C21–C26（数值口径：稳定成立）
全部 bit=False、max_abs = 1–2 bf16 ulp（0.00098/0.0078），mere 值依赖波动
（C21 昨 6.3e-6 → 今 7.8e-3，max_abs 均 1 ulp；核数 24/48 vs 28/56 对照排除分核影响）
→ **ulp 口径判定不变**。

### MLA C04–C20（官方数值对照，新发现：数值大错）
| 组 | mere 范围 | 备注 |
|---|---|---|
| C04–C08/C10/C11（SigA/SigB 典型） | 1.2–2.9 | 远超 ulp |
| C12–C14（r12 长KV 128K–512K） | 9.2–62 | 随 kv 增大 |
| C15–C17（r24） | 27–75 | 同上 |
| C18–C20（r96） | 0.80–0.88 | 头多错小 |

结论：**MLA 路径（rope 分离聚合/update）在 A5 存在真实数值缺陷**（mere 100%+ 量级，
非舍入序差）；此前 C01–C03/C09 的"finite 通过"仅验证可运行性，数值对照后不成立。
kw-4 修复验证集中于 GQA 路径——MLA 路径未获同级修复。

### 性能（卡 4）
- GQA prefill（C21–C26，M≥448）：ratio 1.28–10.6 全亏——M>16 保守分块（kw-4 ③）
  在大 M prefill 形态的循环代价暴露；C23（M=672）10.6× 为最
- S1/S2 decode（昨卡 2 空闲时）净赚 14–17% 与本批不矛盾（T=8 小 M decode 形态）
- MLA 侧性能数据不可信（数值错路径 + 计时抖动，C12/C14/C19 中位异常），修复后需重测

---

## 附 2：kw-8/9 修复后 MLA 战况（2026-09-03，卡 4；证据链 MIGRATION_NOTES P5/P6）

- kw-8（ccd30efc8）：MLA A 装载 V2 泛型 + 15 处 mode-4 同步 → 6/16 ulp
- kw-9（bf0ff4b5d + 79af8275c）：FD lse 缓冲溢出（6144B=24 splits 上限 < AICPU 允许 29）扩 8K
  → **C12–C17 全收敛 1 ulp**；**MLA 11/16 ulp 口径**
- 残差 C05/C06/C07/C10（mere 0.14–0.95）+ C11 偶发 0.0155：根因机制级闭合
  （split 末 S2 宽 ∈(32,128] 时 mm2 A 侧 L1 tile 残留上块 P → 贡献重放，恒正、lse 不变；
  8 组无构建实验含杀块实验），修法 kw-10 进行中
- **性能（收敛后可信）**：MLA 9 个长 KV case 全部快于官方 1.7–9.4×（C14 0.107×）；
  GQA 回归零影响（G1/G2 bit=True 与 kw-4 逐位同值）
- incidental 登记遗留：MLA 路径 V≠K 独立 value 寻址疑坏（absorbed 语义 V=K 不受影响）
- 构建 pitfall：/tmp 被 torch_npu .fed*.so 残留撑满 → TMPDIR 重定向

## 附 3：基线版本勘误（2026-09-03，用户指正）

torch 层 `npu_fused_infer_attention_score_v2`（v2 为 torch API 历史命名）在 A5/CANN 9.1.0 上
经 PTA 动态解析实际调用 **aclnnFusedInferAttentionScoreV5**（日志实锤：
"aclnnFusedInferAttentionScoreV5 is found in .../libcust_opapi.so"；libtorch_npu 引用
V2–V5 全部符号、按特性运行时路由）。底层 kernel 家族名为 FusedInferAttentionScore
（无版本后缀，tiling key 选变体，mix_aic/mix_aiv 双核型）。

三层更正后的基线身份：
| 层 | 名称 |
|---|---|
| torch API | npu_fused_infer_attention_score_v2（接口名 v2） |
| aclnn（实际调度） | **aclnnFusedInferAttentionScoreV5** |
| kernel 家族 | FusedInferAttentionScore_<hash>_mix_{aic,aiv}（ascend950 出厂变体） |

对本报告数据的影响：无——所有 bit/ulp/mere 对照与性能对照的参照系自始至终是这同一个
官方实现（V5），仅文档描述的 aclnn 版本号此前有误（写为 V2）。ulp 残差定性随之更精确：
sink 的 SoftmaxFlashV2（c220 库）vs 官方 V5 的 ProcessVec1Vf MicroAPI 路径舍入序差。

---

## 附 4：kw-11 终态（2026-09-03，C10 攻坚 3 轮未竟，按硬停止线收尾）

- C10 数学定性：ratio 多峰 {10/9, 11/9, 12/9} 等差族（单位质量≈一个 512 块权重，与 b0=8 块预测吻合）；
  D-chunk 选择性二值污染（c2/c3 脏、c0/c1 永净）——覆写/污染类而非计数类
- 探针实锤：污染在 **mm2 累计器本体**（tile 本身 c2/c3 偏高 +35-40%，与全局 +11% 定量自洽）；
  FD staging/combine、除数均无罪
- 两修法未中但保留为 hardening（无回归）：FIX_M/M_FIX 握手恢复（MLA cube 曾注释掉 A3 原设计的
  L0C FIX_M 显式等待）+ V2C2 slot 反向边
- **C10 永久遗留**：触发域 = G≤2（vecDealM=16）× 并发核 ≥16 × 每 split ≥2 循环；已排除 9 类假设；
  下一战役建议 = 改 dump 为 mm2ResGm 原子链逐循环增量，一次二分 fixpipe-atomic vs nupdate-atomic
  （探针基建已常驻，metadata[601] 魔数激活）
- 终态：MLA **15/16 ulp** + GQA bit/ulp 零回退；提交 409de1ac5（探针）/3097c2886（FIX_M）/
  3e4e36a4e（V2C2）；证据链 MIGRATION_NOTES §P6–P6.6
