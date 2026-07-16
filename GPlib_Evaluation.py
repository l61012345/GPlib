
'''
2026-3-5
主要是GP的多级缓存evaluation和一个标准的交叉函数
2026-3-6
移走标准交叉函数
2026-7-16
重构缓存逻辑
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

_MISSING = object()
# 通用LRU缓存类
class LRUCache:
    """轻量级 LRU 缓存，支持 numpy 数组 key 自动清理"""
    def __init__(self, maxsize):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.cache)

    def get_by_key(self, key):
        """
        使用已经构造好的 key 查询缓存。
        命中时将该条目移动到末尾，表示最近使用；
        未命中时返回唯一 sentinel：_MISSING。
        """
        value = self.cache.get(key, _MISSING)
        if value is not _MISSING:
            self.cache.move_to_end(key)
        return value

    def put_by_key(self, key, value):
        """
        使用已经构造好的 key 写入缓存。
        如果 key 已存在，则更新 value 并移动到末尾；
        超过容量时删除最久未使用的条目。
        """
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

def normalize_node_output(result, n_samples):
    """将节点输出统一为连续的 float ndarray"""
    if np.ndim(result) == 0:
        return np.full(n_samples, float(result), dtype=float)

    result = np.asarray(result, dtype=float)
    if not result.flags["C_CONTIGUOUS"]:
        result = np.ascontiguousarray(result)
    return result


def compile_tree(expr, pset, x, prefix="ARG", overflow_inf=True,record_all=False):
    """
    高性能多级缓存版 GP 表达式计算函数。
    ------------------------------------------------
    特性：
    - 使用 L1/L2 多级 LRU 缓存（支持 numpy 数组参数）；
    - 每个节点输出只计算一次 result_key，并向父节点传播；
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

    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy.ndarray")

    n_samples = x.shape[0]
    stack = deque()
    all_outputs = [None] * len(expr) if record_all else None

    # 用于在同一棵树中复用相同 terminal 的输出和 result_key
    terminal_cache = {}

    for node_id, node in enumerate(expr):
        stack.append((node, [], [], node_id))  # (节点, 参数值, 参数key, 节点id)

        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args, arg_keys, node_id = stack.pop()
            result = None
            result_key = None
            func_name = None

            if isinstance(prim, gp.Primitive):
                func_name = prim.name
                func = pset.context[func_name]

                # 直接使用子节点传播上来的 result_key 构造缓存key，
                # 不再重新扫描和哈希实际的numpy数组
                cache_key = (func_name, *arg_keys)

                # === 1. L1 ===
                cached = _L1_cache.get_by_key(cache_key)
                if cached is not _MISSING:
                    result, result_key = cached
                else:
                    # === 2. L2 ===
                    cached = _L2_cache.get_by_key(cache_key)
                    if cached is not _MISSING:
                        result, result_key = cached
                        # L2命中后提升到L1
                        _L1_cache.put_by_key(cache_key, cached)
                    else:
                        # === 3. 真计算 ===
                        computation_succeeded = False

                        try:
                            result = func(*args)
                            result = normalize_node_output(result, n_samples)

                            # 新输出只在第一次产生时计算一次result_key
                            result_key = fast_array_key(result)
                            computation_succeeded = True

                        except OverflowError:
                            if overflow_inf:
                                result = np.full(n_samples, np.nan, dtype=float)
                                result_key = fast_array_key(result)
                                warnings.warn(OverflowError("Overflow happens"))
                            else:
                                # 直接返回第一个参数，同时复用第一个参数的result_key
                                result = args[0]
                                result_key = arg_keys[0]

                        except Exception as error:
                            print(f"[ERROR] {error}")
                            print(f"[ERROR] result: {result}, errorpart: {func_name}")
                            result = np.full(n_samples, np.nan, dtype=float)
                            result_key = fast_array_key(result)

                        # 只有正常计算完成的结果才写入缓存
                        if computation_succeeded:
                            cached = (result, result_key)
                            _L2_cache.put_by_key(cache_key, cached)
                            _L1_cache.put_by_key(cache_key, cached)

            elif isinstance(prim, (gp.Terminal, gp.MetaEphemeral)):
                func_name = prim.name

                if prefix in prim.name:
                    if x.ndim == 1:
                        terminal_identity = ("ARG", 0)
                    else:
                        idx = int(prim.name.replace(prefix, ""))
                        terminal_identity = ("ARG", idx)
                else:
                    value = float(prim.value)
                    terminal_identity = ("CONST", value)

                # 同一棵树中相同terminal直接复用输出和result_key
                cached_terminal = terminal_cache.get(terminal_identity, _MISSING)

                if cached_terminal is not _MISSING:
                    result, result_key = cached_terminal
                else:
                    if terminal_identity[0] == "ARG":
                        if x.ndim == 1:
                            result = x
                        else:
                            idx = terminal_identity[1]
                            result = x[:, idx]
                    else:
                        value = terminal_identity[1]
                        result = np.full(n_samples, value, dtype=float)

                    result = normalize_node_output(result, n_samples)
                    result_key = fast_array_key(result)

                    terminal_cache[terminal_identity] = (result, result_key)

            else:
                raise Exception("Unsupported primitive type!")

            # === 记录节点输出 ===
            if record_all:
                all_outputs[node_id] = result

            # 栈空则返回结果
            if not stack:
                break

            # 同时向父节点传播实际输出和对应的result_key
            stack[-1][1].append(result)
            stack[-1][2].append(result_key)
    return all_outputs if record_all else result


