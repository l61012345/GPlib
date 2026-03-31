__type__ = object
from collections import defaultdict
import functools
import random
import warnings

import numpy as np


def stdcxOnePoint(ind1, ind2,return_indices=False):
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = list(range(0, len(ind1))) # DEAP中的根节点不会选择，这里range改为0以贴合
        types2[__type__] = list(range(0, len(ind2)))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[0:], 0): # STGP中根节点也能被选择
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[0:], 0): # STGP中根节点也能被选择
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    if return_indices:
        return ind1, ind2, index1, index2
    else:
        return ind1, ind2

def mutUniform(individual, expr, pset):
    # Deap原版的mutUniform
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,

def del_indiv_attrs(obj, *attr_paths, warn=False):
    """
    删除对象上的多个属性。

    支持嵌套路径，如果是嵌套属性只删除最后一层，
    如 'fitness.values' 只删除 'values'，保留 'fitness' 对象。

    Parameters
    ----------
    obj : object
    attr_paths : str
        属性路径，如 "fitness.values", "_zero_gate_events"
    warn : bool, default=False
        是否在路径不存在时发出 warning（默认关闭，避免污染日志）

    示例：
        del_indiv_attrs(ind, "fitness.values", "_zero_gate_events")
    """
    for path in attr_paths:
        parts = path.split(".")
        target = obj

        # 逐层定位
        for p in parts[:-1]:
            if not hasattr(target, p):
                if warn:
                    warnings.warn(f"[del_indiv_attrs] Path '{path}' not found. Skipping.")
                target = None
                break
            target = getattr(target, p)

        if target is None:
            continue

        # 删除最后一层
        if hasattr(target, parts[-1]):
            delattr(target, parts[-1])
        else:
            if warn:
                warnings.warn(f"[del_indiv_attrs] Attribute '{path}' not found. Skipping.")

def improvement_tracker(eval_func, *, assign_fitness=False):
    """
    装饰器：计算遗传变异操作（变异或交叉）前后个体的fitness差值。
    对于变异，delta = child_fitness - parent_fitness
    对于返回多个亲本和多个后代的操作，delta = max(child_fitness) - max(parent_fitness)
    assign_fitness参数控制是否将新计算的fitness值直接赋给后代个体的fitness属性。

    返回:
        offspring, delta
    """
    def decorator(op_func):
        @functools.wraps(op_func)
        def wrapper(*parents, **kwargs):
            # parent fitness
            parent_fits = [
                p.fitness.values[0] if p.fitness.valid else None
                for p in parents
            ]

            # apply operator
            offspring = op_func(*parents, **kwargs)
            if not isinstance(offspring, tuple):
                raise ValueError(f"{op_func.__name__} must return a tuple")

            # 重新算child fitness
            child_fits = []
            for c in offspring:
                fit = eval_func(c)
                if assign_fitness:
                    c.fitness.values = fit
                if isinstance(fit, tuple) and len(fit) == 1:
                    fit = fit[0]  # 适应度函数返回单值时，取第一个元素
                else:
                    fit_temp = fit
                    fit = max(fit) if isinstance(fit, tuple) else fit  # 多值时取最大值
                    warnings.warn(
                                    f"Got fitness {fit_temp}, expected single-objective tuple. "
                                    f"Using max(fit)={fit} for tracking ONLY.",
                                    RuntimeWarning
                                )
                child_fits.append(fit)

            # 计算亲本和后代的fitness差
            if len(parents) == 1 and len(offspring) == 1:
                # mutation
                delta = None if parent_fits[0] is None else (child_fits[0] - parent_fits[0])

            elif len(parents) >= 2:
                # crossover
                delta = None if any(f is None for f in parent_fits) \
                    else (max(child_fits) - max(parent_fits))

            else:
                raise ValueError(
                    f"Unsupported arity: {len(parents)} parents, {len(offspring)} offspring"
                )

            return offspring, delta
        return wrapper
    return decorator

def save_rng_state():
    """
    保存 Python random 和 numpy.random 的当前随机状态。
    返回一个可直接保存在变量里的状态字典。
    """
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }


def restore_rng_state(state):
    """
    恢复 Python random 和 numpy.random 的随机状态。
    """
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])