
'''
2026-3-5
主要是GP的多级缓存evaluation和一个标准的交叉函数
2026-3-6
移走标准交叉函数
'''
import numpy as np
from deap import gp
import warnings
from collections import deque, OrderedDict
import hashlib


# 快速哈希函数（取代 md5）
def fast_array_key(a: np.ndarray):
    """
    为 numpy 数组生成快速可哈希 key。
    - 使用 Python 内置 hash(a.tobytes()) + 形状 + dtype
    - 比 md5 快约 10~20x，冲突概率极低
    """
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return (hash(a.tobytes()), a.shape, a.dtype.str)



def hash_output_array(arr):
    """
    将 numpy 数组转为稳定哈希（MD5，节省空间）。
    返回短哈希字符串（前12位）。
    """
    try:
        a = np.asarray(arr, dtype=float)
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        md5 = hashlib.md5(a.tobytes()).hexdigest()
        return md5
    except Exception:
        return "nan"

# 通用LRU缓存类
class LRUCache:
    """轻量级 LRU 缓存，支持 numpy 数组 key 自动清理"""
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def _make_key(self, func_name, args_tuple):
        key_parts = [func_name]
        for arg in args_tuple:
            if isinstance(arg, np.ndarray):
                key_parts.append(fast_array_key(arg))
            else:
                key_parts.append(arg)
        return tuple(key_parts)

    def get(self, func_name, args_tuple):
        key = self._make_key(func_name, args_tuple)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, func_name, args_tuple, value):
        key = self._make_key(func_name, args_tuple)
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)  # 淘汰最旧

    def clear(self):
        self.cache.clear()

    def info(self):
        return {"cache_size": len(self.cache), "maxsize": self.maxsize}


# 全局变量
_global_pset = None

# 全局缓存层

_L1_cache = None  # 中等、代内级
_L2_cache = None  # 最大、全局级

def set_cache_pset(pset, L1_size=10000, L2_size=200000):
    """
    初始化 PrimitiveSet 与多级缓存。
    ------------------------------------------------
    参数：
    - pset: GP 的 PrimitiveSet
    - L1_size: L1 缓存容量（中等）
    - L2_size: L2 缓存容量（较大，全局）
    ------------------------------------------------
    """
    global _global_pset, _L1_cache, _L2_cache
    _global_pset = pset
    _L1_cache = LRUCache(maxsize=L1_size)
    _L2_cache = LRUCache(maxsize=L2_size)



def clear_cache(level=None):
    """
    清空缓存。
    ------------------------------------------------
    参数：
    - level=None: 清空全部缓存；
    - level='L1': 仅清空 L1 缓存；
    - level='L2': 仅清空 L2 缓存。
    ------------------------------------------------
    """
    global _L1_cache, _L2_cache

    if level is None:
        if _L1_cache is not None:
            _L1_cache.clear()
        if _L2_cache is not None:
            _L2_cache.clear()
    elif level == "L1":
        if _L1_cache is not None:
            _L1_cache.clear()
    elif level == "L2":
        if _L2_cache is not None:
            _L2_cache.clear()
    else:
        raise ValueError(f"Invalid cache level '{level}'. Expected one of: None, 'L1', 'L2'.")


def cache_info(level=None):
    """
    返回缓存状态信息。
    ------------------------------------------------
    参数：
    - level=None: 返回所有缓存状态；
    - level='L1': 仅返回 L1 缓存信息；
    - level='L2': 仅返回 L2 缓存信息。
    ------------------------------------------------
    返回：
    - dict 类型，包含缓存大小与上限信息。
    ------------------------------------------------
    """
    global _L1_cache, _L2_cache

    if level is None:
        return {
            "L1": _L1_cache.info() if _L1_cache else {},
            "L2": _L2_cache.info() if _L2_cache else {},
        }
    elif level == "L1":
        return _L1_cache.info() if _L1_cache else {}
    elif level == "L2":
        return _L2_cache.info() if _L2_cache else {}
    else:
        raise ValueError(f"Invalid cache level '{level}'. Expected one of: None, 'L1', 'L2'.")

def compile_tree(expr, pset, x, prefix="ARG", overflow_inf=True,record_all=False):
    """
    高性能多级缓存版 GP 表达式计算函数。
    ------------------------------------------------
    特性：
    - 使用 L1/L2 多级 LRU 缓存（支持 numpy 数组参数）；
    - 自动清理旧缓存，防止内存膨胀；
    - 无锁、单线程安全；
    - 支持溢出保护；
    - 缓存跨个体、跨代共享；
    - 不清理非法输出（NaN / Inf 将被保留）。
    ------------------------------------------------
    参数：
    - expr: gp.PrimitiveTree 对象
    - pset: 当前使用的 PrimitiveSet
    - x: 输入变量矩阵（numpy.ndarray）
    - prefix: 输入变量前缀（默认 "ARG"）
    - overflow_inf: 溢出时返回 np.nan（True）或第一个参数（False）
    - record_all：计算每个节点的输出，为True的时候会返回每个节点的输出，为False的时候只返回个体的输出
    ------------------------------------------------
    返回：
    - 该表达式在 x 上的输出值（numpy.ndarray）
    """
    global _global_pset, _L1_cache, _L2_cache
    if _global_pset is None:
        _global_pset = pset
    if _L1_cache is None:
        _L1_cache = LRUCache(maxsize=2000)
    if _L2_cache is None:
        _L2_cache = LRUCache(maxsize=10000)

    stack = deque()
    all_outputs = [None] * len(expr) if record_all else None

    for node_id, node in enumerate(expr):
        stack.append((node, [], [], node_id))  # (节点, 参数值, 参数名, 节点id)

        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args, arg_names, node_id = stack.pop()
            result = None
            func_name = None

            if isinstance(prim, gp.Primitive):
                func_name = prim.name
                func = pset.context[func_name]
                args_tuple = tuple(args)

                try:
                    # === 1. L1 ===
                    result = _L1_cache.get(func_name, args_tuple)
                    if result is None:
                        # === 2. L2 ===
                        result = _L2_cache.get(func_name, args_tuple)
                        if result is None:
                            # === 3. 真计算 ===
                            result = func(*args)
                            _L2_cache.put(func_name, args_tuple, result)
                        _L1_cache.put(func_name, args_tuple, result)

                except OverflowError:
                    if overflow_inf:
                        result = np.full(x.shape[0], np.nan)
                        warnings.warn(OverflowError("Overflow happens"))
                    else:
                        result = args[0]
                except Exception as error:
                    print(f"[ERROR] {error}")
                    print(f"[ERROR] result: {result}, errorpart: {func_name}")
                    result = np.full(x.shape[0], np.nan)

            elif isinstance(prim, (gp.Terminal, gp.MetaEphemeral)):
                func_name = prim.name
                if prefix in prim.name:
                    if isinstance(x, np.ndarray):
                        if x.ndim == 1:
                            result = x
                        else:
                            idx = int(prim.name.replace(prefix, ""))
                            result = x[:, idx]
                    else:
                        raise ValueError("x must be a numpy.ndarray")
                else:
                    result = np.full(x.shape[0], float(prim.value))
            else:
                raise Exception("Unsupported primitive type!")

            # === 记录节点输出 ===
            if record_all:
                if np.ndim(result) == 0:
                    result = np.full(x.shape[0], float(result)) # 广播常数为向量
                else:
                    result = np.asarray(result, dtype=float)
                all_outputs[node_id] = result

            # 栈空则返回结果
            if not stack:
                break
            stack[-1][1].append(result)
            stack[-1][2].append(func_name)
    return all_outputs if record_all else result


