# a5_port_base 补充包——门禁自包含重建（2026-09-01）

解决 v1 包的两个缺口：
1. 门禁脚本缺 helper（gqa_case_builder/case_builder/fia_v2_cases.csv）
2. 脚本头部硬编码作者环境绝对路径

## 内容
- tests_pr15336/：自包含门禁脚本集（含随包 helper）
  - pr15336_gate.py：G1–G5 门禁（已改脚本相对路径解析，FIA_GATE_TREE 可覆盖）
  - pr15336_gate_c01.py：C01 MLA 门禁（已改自包含，不依赖外部路径）
  - pr15336_graph_probe.py：图模式裁决探针
  - _bootstrap.py：ExtensionFileLoader 引导（绕 PEP 660 editable 拦截）
  - gqa_case_builder.py / case_builder.py / fia_v2_cases.csv：随包 helper/数据
- 门禁用例规格_对接收侧.md：G1–G7 精确参数与验收口径

## 用法（解压后）
cd tests_pr15336
# FIA_GATE_TREE 指向被测 OPP 树（默认=包根/vllm-ascend，可用 env 覆盖）
python3 pr15336_gate.py mqa_c27    # 须 bit_exact=True
python3 pr15336_gate.py kvlen_32768 # 须 bit_exact=True
python3 pr15336_gate_c01.py         # C01 MLA（finite + 口径判读）
## 覆盖项
- FIA_GATE_TREE 未设时默认取脚本同级 ../vllm-ascend/
- helper 模块仅依赖 torch/torch_npu，无外部路径
