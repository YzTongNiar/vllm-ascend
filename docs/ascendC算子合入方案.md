可以。**如果目标是“先服务 vLLM-Ascend/K3 项目，不急着商发进 CANN/torch_npu”**，把 Custom Op 直接放进 `vllm-ascend` 仓里是完全合理的，而且我认为这是当前最自然的交付方式之一。

但要区分：

> **“源码放进 vllm-ascend”可以；“随便丢一个 `.so` 进去然后直接调用”不够工程化。**

## 推荐结构

可以做成类似：

```text
vllm-ascend/
├── vllm_ascend/
│   ├── ops/
│   │   └── grouped_matmul_situ_quant.py
│   └── ...
│
├── csrc/
│   └── grouped_matmul_situ_quant/
│       ├── kernel/
│       │   ├── ...
│       │   └── gmm_situ_quant.cpp
│       ├── host/
│       │   ├── tiling.cpp
│       │   └── register.cpp
│       ├── include/
│       └── CMakeLists.txt
│
├── tests/
│   └── ops/
│       └── test_grouped_matmul_situ_quant.py
│
└── setup.py / pyproject.toml
```

构建安装：

```text
pip install vllm-ascend
        ↓
同时编译 custom op
        ↓
libgmm_situ_quant.so
        ↓
随 vllm-ascend package 一起安装
```

运行时由 `vllm_ascend` 自动加载：

```python
torch.ops.load_library(...)
```

然后上层不要到处直接写：

```python
torch.ops.npu.gmm_situ_quant_weight_nz(...)
```

而是统一封一层：

```python
from vllm_ascend.ops.grouped_matmul_situ_quant import (
    grouped_matmul_situ_quant,
)

y, y_scale = grouped_matmul_situ_quant(...)
```

这样更合适。

---

# 为什么放 vLLM-Ascend 很合理

因为这个算子目前的需求来源就是：

```text
K3 / MoE
    ↓
vLLM-Ascend 推理
    ↓
GMM1 + DequantSituQuant / SituMxQuant
    ↓
希望融合
```

它现在不是一个已经证明：

```text
所有 Ascend PyTorch 用户
所有模型
所有框架
```

都需要的公共 primitive。

因此归属在：

```text
vLLM-Ascend
```

可以理解成：

> **模型推理框架为了获得性能而携带自己的设备特化 kernel。**

这在 AI Infra 里非常正常。

---

# 调用链就会变得很简单

现在：

```text
K3 model implementation
        │
        ▼
vllm_ascend.ops.grouped_matmul_situ_quant()
        │
        ▼
torch.ops.npu.gmm_situ_quant_weight_nz()
        │
        ▼
vLLM-Ascend 自带 Custom Op .so
        │
        ├── Host tiling
        └── AscendC kernel
                │
                ▼
               NPU
```

注意这里仍然会经过 **PyTorch Dispatcher**。

只是你不需要：

```text
torch_npu 新增正式 API
```

也不需要：

```text
CANN 新增 aclnnGroupedMatmulSituQuant
```

。

---

# 但我建议不要让模型代码直接依赖 `torch.ops`

这是一个很重要的工程边界。

不推荐：

```python
# model.py 到处这样写
torch.ops.npu.gmm_situ_quant_weight_nz(...)
```

推荐：

```text
Model
  ↓
vllm_ascend.ops.gmm_situ_quant()
  ↓
torch.ops.npu.xxx
```

因为未来很可能发生：

### 今天

```python
torch.ops.npu.gmm_situ_quant(...)
```

### 以后进入 torch_npu

```python
torch_npu.npu_gmm_situ_quant(...)
```

### 再以后进入 CANN

```text
torch_npu
  ↓
aclnnGroupedMatmulSituQuant
```

只要有 wrapper：

```python
def grouped_matmul_situ_quant(...):
    ...
```

模型侧完全不用改。

---

# 你们当前 A3/A5 甚至可以共用这个接口层

例如：

```python
def grouped_matmul_situ_quant(
    x,
    weight,
    x_scale,
    weight_scale,
    group_list,
    ...,
):
    if is_a5():
        return torch.ops.npu.gmm_situ_quant_a5(...)
    elif is_a3():
        return torch.ops.npu.gmm_situ_quant_a3(...)
```

或者 kernel 内部/Host tiling 按 SoC 分发：

```text
GroupedMatmulSituQuant
        │
        ├── Ascend910_93 / A3
        │      production W4A8 GMM
        │      + DequantSituQuant
        │
        └── Ascend950 / A5
               production A8W4-MX GMM
               + SituMxQuant
```

对 vLLM-Ascend 上层来说仍然是一个语义接口。

---

# `.so` 应该怎么处理

这里有两个方案。

## 方案一：源码随 vLLM-Ascend 编译

我更推荐。

```text
vllm-ascend repo
   ├── AscendC source
   ├── host source
   └── build system

pip install .
       ↓
编译对应 SoC custom op
```

优点：

* 源码和框架版本绑定；
* CI 可以直接测；
* CANN 版本适配清晰；
* 不需要手工复制 `.so`；
* A3/A5 可以分别编译。

---

## 方案二：预编译 `.so` 放仓里

例如：

```text
vllm_ascend/lib/libgmm_situ_quant.so
```

也能用，但长期不太推荐。

因为 `.so` 会绑定：

```text
CANN version
torch_npu ABI
Python
SoC
compiler
```

例如：

```text
CANN 9.1
```

编出来的东西，以后升级 CANN 时未必安全。

所以正式接入 vLLM-Ascend 更适合：

> **源码进入仓库，安装阶段编译 `.so`。**

预编译 `.so` 可以作为当前项目快速交付物。

---

# 还有一个值得提前想清楚的问题：vendored CANN GMM 源码

你们现在准备：

```text
CANN production GroupedMatmul source
        ↓
vendoring
        ↓
修改 SiTU epilogue
        ↓
Custom Op
```

如果最终把它放到 vLLM-Ascend 仓库，要特别处理：

```text
license
源码来源
CANN version
upstream path
修改说明
```

最好目录里明确：

```text
csrc/grouped_matmul_situ_quant/
├── vendor/
│   └── cann_9_1/
├── epilogue/
└── README.md
```

README 写清：

```text
Upstream:
CANN 9.1.0
opp/built-in/op_impl/.../grouped_matmul/...

Local modifications:
- retain production GMM pipeline
- replace output path
- fuse SiTU
- fuse dynamic quant
```

这样以后 CANN 9.2/10.x 更新 GMM 时，才知道怎么 rebase。

---

# 和“ops-transformer 交付”相比有什么区别

如果放在 vLLM-Ascend：

```text
所有权：
vLLM-Ascend

服务对象：
vLLM-Ascend

发布周期：
跟 vLLM-Ascend
```

如果放 ops-transformer/CANN：

```text
所有权：
CANN 算子团队

服务对象：
所有上层框架

发布周期：
跟 CANN
```

因此现在：

```text
性能特性仍处于快速迭代
+
明确由 K3/vLLM-Ascend 驱动
```

放 vLLM-Ascend 更敏捷。

以后证明：

```text
DeepSeek / GLM / K3 / 更多 MoE
都需要这个 primitive
```

再下沉到 CANN 才合理。

---

## 我会建议你们当前这样定

```text
当前项目交付：

vLLM-Ascend
│
├── Python wrapper
│
├── GroupedMatmulSituQuant Custom Op
│   ├── A3
│   └── A5
│
├── build/install
│
└── correctness/perf tests
```

底层实现：

```text
A3:
production A3 GMM source
+ DequantSituQuant

A5:
production A5 GMM source
+ SituMxQuant
```

未来演进：

```text
vLLM-Ascend Custom Op
        │
        │ 接口成熟 / 多项目复用
        ▼
CANN Built-in
        │
        ▼
torch_npu 正式 API
```

所以答案是：**可以，而且对当前阶段很合适。**只是最好把它作为 `vllm-ascend` 的正式 custom-op 子模块来构建、加载和测试，而不是简单把一个二进制 `.so` 扔进去。
