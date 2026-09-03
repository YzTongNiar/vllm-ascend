# 权威源切换声明（variant-1 迁移，2026-08-31）

## 结论

**自即日起，FIA v2 Sink 双算子的权威源与构建载体切换为：**

```
/home/t00886357/npu_fused_infer_attention_score_v2_sink/vllm-ascend/
    └── csrc/attention/fused_infer_attention_score_v2_sink{,_metadata,_common}/
```

基线 = PR #15336 分支 @ `34f94ca34`（feat(attention): use device-side FIA tiling），
该树携带全部 round_4/5 修复（血缘审计 + 四标记独立复核 + 迁移后门禁回归全绿）。

## 迁移后迭代规则

1. **算子/集成层改动**一律在 `vllm-ascend/csrc/attention/fused_infer_attention_score_v2_sink*/`
   与 `vllm_ascend/ops/fia_v2_sink.py` 上进行；构建用根目录 `build.sh`（F2：--fast 增量 / 默认全量）。
2. **`op_src/` 转为存档只读**（provenance 留档：round_4/5 修复的权威出处与 diff 基线 =
   `results/round_4_d192/rollback/` 之前的形态 + `handoff/round4_patches/`）。
   禁止再在 op_src 上做新改动（双头修改 = 漂移）。
3. **上游跟踪**：PR #15336 合入/演进后，按 vendored README 的 upstream 声明对账，
   以 git patch 序列跟移（git 历史随私仓 clone 保留）。
4. **门禁**：`tests/pr15336/pr15336_gate{,_c01}.py` / `pr15336_graph_probe.py`
   （`FIA_GATE_TREE` 环境变量可指向任意树，默认本仓 vendored 树）+
   既有 `tests/gqa_case_builder.py` / `check_gqa.py` / `case_builder.py`（C01–C20）。

## 迁移验证记录（2026-08-31）

vendored 树（含 build 产物原样拷贝）冒烟门禁：mqa_c27 / mqa_c28 / C21 / kvlen_4096 /
b2_248 全部 bit 级一致 ✓（与 PR 原构建结果逐项一致，拷贝无损）。
`build.sh` 全量重建冒烟待下轮排程（预期无损；clean-build 纪律已写入脚本默认行为）。
