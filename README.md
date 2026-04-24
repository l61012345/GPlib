# GPlib Utilities for DEAP Genetic Programming  

# 用于 DEAP 遗传编程的 GPlib 工具集

A lightweight utility collection for building, evaluating, varying, and monitoring Genetic Programming (GP) experiments with [DEAP](https://deap.readthedocs.io/).

一个轻量级工具集，用于基于 [DEAP](https://deap.readthedocs.io/) 构建、评估、变异和监控遗传编程（Genetic Programming, GP）实验。

This repository currently contains three modules:

本仓库目前包含三个模块：

- `GPlib_Evaluation.py` — fast GP tree evaluation with multi-level LRU caching.  
  `GPlib_Evaluation.py` —— 带有多级 LRU 缓存的快速 GP 树评估工具。

- `GPlib_GeneticVariations.py` — DEAP-compatible genetic operators and experiment utilities.  
  `GPlib_GeneticVariations.py` —— 与 DEAP 兼容的遗传操作算子和实验辅助工具。

- `GPlib_Graphs.py` — training-curve trackers for fitness, tree size, and custom metrics.  
  `GPlib_Graphs.py` —— 用于记录 fitness、树大小和自定义指标的训练曲线追踪器。

The code is designed for symbolic regression and GP-style evolutionary experiments where repeated subtree evaluation, reproducibility, and visual monitoring are important.

该代码主要面向符号回归和 GP 类演化实验，尤其适合需要重复子树评估、随机性复现和实验过程可视化监控的场景。

---

## Features  

## 功能概览

### 1. Cached GP tree evaluation  

### 1. 带缓存的 GP 树评估

`GPlib_Evaluation.py` provides a high-performance replacement for repeated GP expression evaluation.

`GPlib_Evaluation.py` 提供了一个高性能的 GP 表达式评估工具，可用于替代重复调用式的树表达式计算。

Key features:

主要功能：

- L1/L2 multi-level LRU cache for primitive outputs.  
  对 primitive 输出使用 L1/L2 多级 LRU 缓存。

- NumPy-array-aware cache keys.  
  支持 NumPy 数组输入的缓存 key 生成。

- Optional recording of every node output in a tree.  
  可选择记录 GP 树中每个节点的输出。

- Overflow handling for unsafe numerical primitives.  
  对可能发生数值溢出的 primitive 提供异常处理。

- Cache inspection and manual clearing utilities.  
  提供缓存状态查看和手动清理工具。

Main functions/classes:

主要函数/类：

```python
set_cache_pset(pset, L1_size=10000, L2_size=200000)
compile_tree(expr, pset, x, prefix="ARG", overflow_inf=True, record_all=False)
clear_cache(level=None)
cache_info(level=None)
LRUCache(maxsize)
```

---

### 2. Genetic variation utilities  

### 2. 遗传变异辅助工具

`GPlib_GeneticVariations.py` provides DEAP-compatible variation operators and helper tools.

`GPlib_GeneticVariations.py` 提供了与 DEAP 兼容的变异/交叉算子以及实验辅助函数。

Key features:

主要功能：

- Standard one-point crossover with optional return of selected crossover indices.  
  标准单点交叉，并可选择返回交叉节点索引。

- Uniform subtree mutation.  
  均匀子树变异。

- Safe deletion of individual attributes, including nested attributes.  
  安全删除个体属性，支持嵌套属性路径。

- Decorator for tracking fitness improvement before and after variation.  
  提供装饰器，用于记录遗传操作前后的 fitness 变化。

- Save and restore Python/NumPy random states for reproducibility.  
  支持保存和恢复 Python `random` 与 NumPy 随机状态，便于实验复现。

Main functions:

主要函数：

```python
stdcxOnePoint(ind1, ind2, return_indices=False)
mutUniform(individual, expr, pset)
del_indiv_attrs(obj, *attr_paths, warn=False)
improvement_tracker(eval_func, *, assign_fitness=False)
save_rng_state()
restore_rng_state(state)
```

---

### 3. GP graph tracking  

### 3. GP 训练曲线追踪

`GPlib_Graphs.py` provides plotting utilities for monitoring GP training.

`GPlib_Graphs.py` 提供了用于监控 GP 训练过程的绘图工具。

Key features:

主要功能：

- Track best fitness, mean fitness, and average tree size.  
  记录最佳 fitness、平均 fitness 和平均树大小。

- Save plots and tracker data automatically.  
  自动保存图像和 tracker 数据。

- Support headless plotting through Matplotlib's `Agg` backend.  
  支持通过 Matplotlib 的 `Agg` 后端进行无界面绘图。

- Adaptive multi-panel tracker for arbitrary metrics.  
  支持自定义指标的多子图自适应 tracker。

- Sparse point annotation for long runs.  
  对长时间实验提供稀疏数值标注，避免标签过密。

Main classes:

主要类：

```python
GraphTracker
AdaptiveGraphTracker
```

---

## Installation  

## 安装

Clone the repository:

克隆本仓库：

```bash
git clone https://github.com/l61012345/GPlib.git
cd YOUR_REPOSITORY
```

Install dependencies:

安装依赖：

```bash
pip install numpy deap matplotlib
```

Optional but recommended:

可选但推荐安装：

```bash
pip install scipy pandas
```

The core modules only require:

核心模块主要依赖：

```text
numpy
deap
matplotlib
```

---

## Quick Start  

## 快速开始

### 1. Cached evaluation  

### 1. 缓存评估

```python
import numpy as np
from deap import gp

from GPlib_Evaluation import set_cache_pset, compile_tree, cache_info, clear_cache

# pset should be a DEAP PrimitiveSet or PrimitiveSetTyped
# pset 应为 DEAP 的 PrimitiveSet 或 PrimitiveSetTyped
set_cache_pset(pset, L1_size=10000, L2_size=200000)

# X: input matrix, shape = (n_samples, n_features)
# X: 输入矩阵，形状为 (样本数, 特征数)
X = np.asarray(X, dtype=float)

# individual: DEAP PrimitiveTree
# individual: DEAP PrimitiveTree 个体
y_pred = compile_tree(
    expr=individual,
    pset=pset,
    x=X,
    prefix="ARG",
    overflow_inf=True,
    record_all=False,
)

print(cache_info())

# Clear only generation-level cache if needed
# 如有需要，可只清理 L1 缓存
clear_cache("L1")
```

To record the output of every node:

如需记录每个节点的输出：

```python
node_outputs = compile_tree(
    expr=individual,
    pset=pset,
    x=X,
    record_all=True,
)

root_output = node_outputs[0]
```

---

### 2. Genetic variation  

### 2. 遗传变异

```python
from GPlib_GeneticVariations import stdcxOnePoint, mutUniform

# One-point crossover
# 单点交叉
child1, child2 = stdcxOnePoint(parent1, parent2)

# One-point crossover with selected node indices
# 单点交叉，并返回被选中的节点索引
child1, child2, idx1, idx2 = stdcxOnePoint(
    parent1,
    parent2,
    return_indices=True,
)

# Uniform subtree mutation
# 均匀子树变异
mutant, = mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
```

Track operator improvement:

追踪遗传操作带来的 fitness 变化：

```python
from GPlib_GeneticVariations import improvement_tracker

tracked_mutation = improvement_tracker(
    eval_func=toolbox.evaluate,
    assign_fitness=False,
)(mutUniform)

offspring, delta = tracked_mutation(individual, expr=toolbox.expr_mut, pset=pset)

print("Fitness improvement:", delta)
```

Save and restore random state:

保存和恢复随机状态：

```python
from GPlib_GeneticVariations import save_rng_state, restore_rng_state

state = save_rng_state()

# Run stochastic operations here
# 在这里执行随机操作

restore_rng_state(state)
```

---

### 3. Plot GP training curves  

### 3. 绘制 GP 训练曲线

```python
from GPlib_Graphs import GraphTracker

tracker = GraphTracker(
    LiveDisplay=False,
    filename="results/gp_training_curve",
    dpi=550,
    format="tiff",
)

for gen in range(n_generations):
    # evolve population ...
    # 执行一代演化 ...
    tracker.update(gen, population)
    tracker.plot()
```

The tracker saves:

该 tracker 会保存：

```text
results/gp_training_curve.tiff
results/gp_training_curve.pkl
```

---

### 4. Adaptive tracker for custom metrics  

### 4. 自定义指标的 Adaptive Tracker

```python
from GPlib_Graphs import AdaptiveGraphTracker

tracker = AdaptiveGraphTracker(
    tracked_layout=[
        ["best_fitness", "mean_fitness"],
        "mean_size",
        ["tie_score_mean", "tie_score_var"],
        "potential_count",
    ],
    LiveDisplay=False,
    filename="results/adaptive_training_curve",
    dpi=550,
    format="tiff",
    name_map={
        "best_fitness": "Best Fitness",
        "mean_fitness": "Mean Fitness",
        "mean_size": "Mean Tree Size",
        "tie_score_mean": "Tie-score Mean",
        "tie_score_var": "Tie-score Variance",
        "potential_count": "Potential Count",
    },
    fmt_map={
        "best_fitness": "{:.4f}",
        "mean_fitness": "{:.4f}",
        "mean_size": "{:.1f}",
        "potential_count": "{:.0f}",
    },
)

for gen in range(n_generations):
    stats = {
        "best_fitness": best_fit,
        "mean_fitness": mean_fit,
        "mean_size": mean_size,
        "tie_score_mean": tie_mean,
        "tie_score_var": tie_var,
        "potential_count": potential_count,
    }

    tracker.update_from_dict(gen, stats)
    tracker.plot()
```

---

## Recommended Repository Structure  

## 推荐仓库结构

```text
.
├── GPlib_Evaluation.py
├── GPlib_GeneticVariations.py
├── GPlib_Graphs.py
├── README.md
├── LICENSE
└── examples/
    └── basic_symbolic_regression.py
```

---

## Notes and Limitations  

## 注意事项与限制

1. The cached evaluator assumes that GP primitives are deterministic for the same inputs.  
   缓存评估器假设 GP primitive 在相同输入下是确定性的。

2. Cache keys for NumPy arrays are designed for speed. For strict cross-process or long-term persistent hashing, use a stable hash such as MD5.  
   NumPy 数组的缓存 key 主要为了速度设计。如果需要跨进程或长期持久化的严格哈希，建议使用 MD5 等稳定哈希方式。

3. The plotting utilities are optimized for experiment monitoring rather than publication-ready figure styling.  
   绘图工具主要用于实验过程监控，而不是直接面向论文发表级别的图像排版。

4. `compile_tree` is intended for DEAP `PrimitiveTree`-style prefix trees.  
   `compile_tree` 主要面向 DEAP `PrimitiveTree` 风格的前缀表达式树。

5. When using multiprocessing, initialize caches carefully inside each worker process if needed.  
   使用多进程时，如有需要，应在每个 worker 进程中谨慎初始化缓存。

---

## Citation  

## 引用

If you use this code in academic work, please cite the associated paper or repository once available.

如果你在学术工作中使用本代码，请在相关论文或仓库信息可用后进行引用。

Suggested repository citation format:

建议的仓库引用格式：

```bibtex
@misc{gplib_utilities,
  title        = {GPlib Utilities for DEAP Genetic Programming},
  author       = {Yilin Liu},
  year         = {2026},
  howpublished = {\url{https://github.com/l61012345/GPlib}},
  note         = {Utility modules for cached evaluation, genetic variation, and graph tracking in DEAP-based Genetic Programming}
}
```

---

## License  

## 开源许可证

This project is released under the MIT License.

本项目采用 MIT License 开源。

You may replace this section with another license if required by your institution or project.
