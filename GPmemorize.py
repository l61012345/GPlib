from collections import deque
import logging
import multiprocessing
import heapq
import numpy as np
from deap import gp

# 将 Manager 和 shared_log 创建放到主进程
def get_shared_log(manager=None):
    if manager is None:
        manager = multiprocessing.Manager()  # 在主进程中创建 Manager
    if not hasattr(get_shared_log, "_manager"):
        get_shared_log._manager = manager
        get_shared_log.shared_log = get_shared_log._manager.dict()
    return get_shared_log.shared_log

def log_decorator(shared_log, expr_str):
    """
    装饰器：包装原语函数，记录函数名称、输入、输出以及形状信息。
    参数:
    - shared_log: 共享日志列表
    - expr_str: 当前调用的表达式字符串（例如 "add(ARG0, ARG1)")
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 记录输入的形状和数值（部分）
            if isinstance(args, (int, float)):
                input_values = tuple(args)
            else:
                input_values = tuple(np.concatenate(args,axis=0))
            # 生成哈希键，唯一标识一个计算
            key = (expr_str, input_values)
            # 检查共享日志
            if key in shared_log:
                shared_log[key]["count"] += 1
                return shared_log[key]["output_value"]  # 直接返回已计算结果
            else:
                # 调用原函数
                result = func(*args, **kwargs)
                shared_log[key] = {
                    "function": expr_str,
                    "input_values": input_values,
                    "output_value": result,
                    "count": 1,
                }
                return result
        return wrapper
    return decorator

def compute_tree(expr, pset,x,prefix="ARG",overflow_inf = True,shared_log=None):
    # 初始化栈，用于递归计算每个节点
    stack = deque()
    # 遍历树中的每个节点并递归计算值
    for id, node in enumerate(expr):
        stack.append((node, [], [], id))  # 将节点和空参数列表压入栈
        while len(stack[-1][1]) == stack[-1][0].arity:  # 确保所有子节点都被处理
            prim, args, arg_expressions, id = stack.pop()  # 获取当前节点的原语和参数
            if isinstance(prim, gp.Primitive):
                # 对于 Primitive 节点，调用相应的原语函数计算结果
                if arg_expressions:
                    expr_str = f"{prim.name}({', '.join(arg_expressions)})" # 如果有拼好的表达式就直接拿来用
                else:
                    expr_str = f"{prim.name}({', '.join(map(str, args))})" # 如果没有就重新创建一个
                if shared_log != None:
                    decorated_func = log_decorator(shared_log, expr_str)(pset.context[prim.name]) # 调用当前的函数
                else:
                    decorated_func = pset.context[prim.name]
                try:
                    result = decorated_func(*args)
                except OverflowError as e:
                    # 对溢出的处理
                    if overflow_inf == True:
                        result = float('inf') # 返回inf
                    else:
                        result = args[0]  # 返回第一个参数作为结果
                    logging.error("Overflow error occurred: %s, args: %s", str(e), str(args))
            elif isinstance(prim, gp.Terminal):
                # 对于 Terminal 节点，获取数据集中的相应特征值
                if prefix in prim.name: # 这是个变量
                    if isinstance(x, (np.ndarray)):
                        expr_str = prim.name
                        if x.ndim == 1:
                            result = x  # 直接使用整个数组
                        else:
                            result = x[:, int(prim.name.replace(prefix, ""))] # 把变量对应的值带入运算
                    else:
                        raise ValueError("Datatype should be np.ndarray")
                else:
                    result = float(prim.value)  # 对于常量终结符，直接使用值
                    expr_str = str(prim.value)
            else:
                raise Exception("Unsupported primitive type!")
            # 将结果传递给父节点（即将当前节点的结果作为参数传递给上层）
            if not stack:
                break  # 如果栈为空，表示所有节点都已经计算过
            stack[-1][1].append(result)  # 将计算结果添加到栈顶父节点的参数列表
            stack[-1][2].append(expr_str)  # 记录子表达式
    # 最终返回树的计算结果
    return result

def find_top_subtree_in_log(topn:int,pset:gp.PrimitiveSet,shared_log:dict,key_name:str,largest = True, verbose = 1):
    '''
    - topn: return top n subtrees in the log
    - pset: gp.PrimitiveSet
    - shared_log: log to be searched
    - key: key to be ranked in shared_log
    - largest: from large to small if True, else from small to large when ranking
    - verbose: 1- print record 0-no print
    '''
    if largest == True:
        top_n_keys = heapq.nlargest(topn, shared_log.keys(), key=lambda k: shared_log[k][key_name])
    else:
        top_n_keys = heapq.nsmallest(topn, shared_log.keys(), key=lambda k: shared_log[k][key_name])
    top_n_subtrees = []
    for rank, key in enumerate(top_n_keys, start=1):
        subtree = key[0]  # 提取 expr_str
        key_value = shared_log[key][key_name] 
        top_n_subtrees.append(gp.PrimitiveTree.from_string(subtree,pset=pset))  # 存入列表
        if verbose == 1:
            print(f"Rank {rank}: Subtree: {subtree}, {key_name}: {key_value}")
    return top_n_subtrees