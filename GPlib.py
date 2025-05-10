import multiprocessing.managers
import numpy as np
import time
from deap import base, creator, tools, gp
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
import GPutilities
import multiprocessing
from collections import deque
import threading
import matplotlib.pyplot as plt
from typing import Callable
lock = multiprocessing.Lock()

class GPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        pset:gp.PrimitiveSet=None,
        pop_size:int=100,
        gen_num:int=20,
        genetic_operator_pipline:__module__=None,
        fitness_function:Callable=None,
        fitness_weight:int=-1,        
        parsimony:float=0.000,
        value_log:multiprocessing.managers.DictProxy=None,
        tracker:bool = False,
        init_mintree_height:int=1,
        init_maxtree_height:int=3,
        hof_size:int=1,
        elitism:bool=False,
        seed:int=None,
        n_jobs:int=1,
        verbose:bool=True,
    ):
        """
        pset: PrimitiveSet, use it as DEAP PrimitiveSet \\
        popsize: int, size of the population \\
        gen_num: int, number of generations \\
        genetic_operator_pipline: class module, user defined pipline of genetic operations, including selection, reproduction, crossover and mutation \\
        fitness function: function \\
        parsimony: float, panelty of size \\
        seed: random seeds \\
        value_log: multiprocessing.Manager.dict(),a dict used during evaluation for memorize subtrees result, use shared_log = manager.dict() to create one. \\
        tracker: bool, if is True, then it will continuely records the change of the subtree with max count for each generation in value_log. 
                 Use model.best_function_dict to access the final result\\
        fitness_weight: the weight of fitness, usually -1 (min is best) or 1 (max is best) \\
        init_mintree_height: the tree min height in initial population \\
        init_maxtree_height: the tree max height in initial population \\
        hof_size: int, the size of hall-of-frame \\
        elitism: bool, flag to open elitism \\
        verbose: bool, open the log or not. \\
        n_jobs: int, num of pool for multiprocessing used
        """
        self.pop_size = pop_size
        self.gen_num = gen_num
        self.pset = pset
        self.genetic_operator_pipline = genetic_operator_pipline
        self.fitness_function = fitness_function
        self.parsimony = parsimony
        self.seed = seed
        self.fitness_weight = fitness_weight
        self.value_log = value_log
        self.init_mintree_height = init_mintree_height
        self.init_maxtree_height = init_maxtree_height
        self.hof_size = hof_size
        self.elitism = elitism
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.tracker = tracker

        if pset is None:
            raise ValueError("pset is empty, you must have one.")
        # 没有value_log的时候提醒创建一个，不然无法加速
        if value_log is None:
            warnings.warn(
                "No value_log, use shared_log = manager.dict() to create one",
                UserWarning)
            self.value_log = None
            time.sleep(1)
        else:
            if self.tracker is True:
                self.best_function_dict = {}
            else:
                self.best_function_dict = None
        if fitness_function is None:
            warnings.warn(
                "No Custom defined fitness function,use MSE instead", UserWarning)
            time.sleep(1)
            fitness_weight = (-1)  # 默认使用MSE作为fitness function的时候，fitness越小越好
        if genetic_operator_pipline is None:
            warnings.warn(
                "No Custom defined Genetic Operations,use default: 3 Tournament, 0.9 std crossover, and 0.2 NodeReplace mutation instead",
                UserWarning)
            time.sleep(1)

        
    def _setup_gp(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            warnings.warn('Random Seed is fixed')
        # 创建fitness
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(self.fitness_weight,))
        # 创建个体
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=self.init_mintree_height,
            max_=self.init_maxtree_height)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def safe_log_write(self, key, entry):
        """线程安全地写入 shared_log，避免不可序列化对象"""
        # 加了个lock来专门更新entry，避免出问题
        with lock:
                self.value_log[key] = entry

    def maybe_clean_log(self, max_entry):
        """按需启动清理线程"""
        if len(self.value_log) > max_entry * 1.2:
            cleaner = threading.Thread(
                target=self.clean_log,
                args=(max_entry,),
                daemon=True  # 后台线程，不阻塞主线程退出
            )
            cleaner.start()

    def clean_log(self, max_entry):
        """保留使用频率最高的 max_entry 项"""
        with lock:
            if len(self.value_log) <= max_entry:
                pass
            else:
                warnings.warn("Clean the log, only keep large count entry")
                # 按使用次数排序，保留前 max_entry 项
                sorted_items = sorted(
                    self.value_log.items(),
                    key=lambda x: x[1].get("count", 0),
                    reverse=True
                )
                self.value_log.clear()
                self.value_log.update(dict(sorted_items[:max_entry]))

    def decorator(self,func,expr_str):
        def wrapper(*args, **kwargs):
            key = expr_str
            #input_str = '|'.join(map(str, args))
            #key = f"{expr_str}|{input_str}"
            with lock:
                if key in self.value_log:
                    # 如果存在，更新 count 值
                    entry = self.value_log[key]
                    entry['count'] += 1
                    output = entry['output_value']
                else:
                    output = func(*args, **kwargs)
                    entry = {
                        'function': expr_str,
                        #'input_values': input_str,
                        'output_value': output,
                        'count': 1
                    }
            self.safe_log_write(key, entry)
            return output
        return wrapper

    def log_decorator(self,expr_str,func):
        # 用于调用包装器的函数
        decorator = self.decorator(func,expr_str)
        return decorator

    def compute_tree(self,expr, pset,x, prefix="ARG",overflow_inf = True,shared_log=None):
        # 初始化栈，用于递归计算每个节点
        stack = deque()
        # 遍历树中的每个节点并递归计算值
        for id, node in enumerate(expr):
            stack.append((node, [], [], id))  # 将节点和空参数列表压入栈
            while len(stack[-1][1]) == stack[-1][0].arity:  # 确保所有子节点都被处理
                prim, args, arg_expressions, id = stack.pop()  # 获取当前节点的原语和参数
                result = None # 清空result
                if isinstance(prim, gp.Primitive):
                    # 对于 Primitive 节点，调用相应的原语函数计算结果
                    if shared_log is not None: # 是否开启了share_log功能
                        if arg_expressions:
                            expr_str = f"{prim.name}({', '.join(arg_expressions)})" # 如果有拼好的表达式就直接拿来用
                        else:
                            expr_str = f"{prim.name}({', '.join(map(str, args))})" # 如果没有就重新创建一个
                        decorated_func = self.log_decorator(expr_str,pset.context[prim.name]) # 调用当前的函数
                    else:
                        expr_str = None
                        decorated_func = pset.context[prim.name] # 没开启就直接调用函数
                    try:
                        result = decorated_func(*args)
                    except OverflowError as e:
                        # 对溢出的处理
                        if overflow_inf == True:
                            result = np.inf # 返回inf
                            warnings.warn(OverflowError("Overflow happens"))
                        else:
                            result = args[0]  # 返回第一个参数作为结果
                    except Exception as error:
                        print('error')
                        print(result)
                        print('errorpart',expr_str)
                        raise error
                        result = float('inf') # 返回inf
                    
                elif isinstance(prim, gp.Terminal) or isinstance(prim,gp.MetaEphemeral):
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

    def eval_func(self,ind,X,y):
        try:
            # 计算树在X上的输出
            pred = self.compute_tree(
                ind, pset=self.pset, x=X, shared_log=self.value_log
            )
            # 如果 pred 是一个常数或标量，广播成与 y 相同形状
            if np.isscalar(pred) or (isinstance(pred, np.ndarray) and pred.size == 1):
                pred = np.full_like(y, fill_value=float(pred))
            if callable(self.fitness_function):
                loss = self.fitness_function(pred, y) # 使用自定义的fitness function进行衡量
            else:
                # 用户没有定义的话就默认用MSE函数
                loss = np.mean((pred - y) ** 2)
            return (loss + self.parsimony * len(ind) * self.fitness_weight,)
        except Exception as e:
            print(ind)
            raise e
            return (float("inf"),)


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X, self.y = X, y  # 确保在类内共享
        self._setup_gp()

        self.toolbox.register("evaluate", self.eval_func,X=X,y=y)

        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(self.hof_size)

        # 初次evaluation
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
                fitnesses = pool.map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # 初始化记录
        pop_fit = tools.Statistics(lambda ind: ind.fitness.values)
        pop_size = tools.Statistics(len)  # Object: Pop size
        ind_height = tools.Statistics(
            lambda ind: ind.height
        )  # Object:individual height
        mstats = tools.MultiStatistics(
            fitness=pop_fit, size=pop_size, height=ind_height
        )
        record = mstats.compile(pop)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        mstats.register("num", len)
        logbook = tools.Logbook()
        logbook.header = (
            "gen",
            "time",
            "num",
            "new",
            "fitness",
            "size",
            "height",
            "best_ind_height",
            "best_ind_len",
            "best_ind",
        )
        logbook.chapters["fitness"].header = "min", "avg", "max"
        logbook.chapters["size"].header = "min", "avg", "max"
        logbook.chapters["height"].header = "min", "avg", "max"

        for g in range(self.gen_num):
            start_time = time.perf_counter()
            if self.genetic_operator_pipline is None:  # 如果用户没有定义pipline，就用默认设置
                # 锦标赛选择
                offspring = tools.selTournament(pop, len(pop), 3)
                offspring = list(map(self.toolbox.clone, offspring))
                # 标准交叉
                for i in range(1, len(offspring), 2):
                    if np.random.rand() < 0.9:
                        offspring[i - 1], offspring[i] = gp.cxOnePoint(
                            offspring[i - 1], offspring[i]
                        )
                        del offspring[i - 1].fitness.values, offspring[i].fitness.values
                # 点突变
                for i in range(len(offspring)):
                    if np.random.rand() < 0.1:
                        (offspring[i],) = gp.mutNodeReplacement(offspring[i],self.pset)
                        del offspring[i].fitness.values
            else:
                offspring = self.genetic_operator_pipline(pset=self.pset).apply(pop)

            # 重新衡量结构有改动的个体
            # 并行评估
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            with multiprocessing.Pool(processes=self.n_jobs) as pool:
                fitnesses = pool.map(self.toolbox.evaluate, invalids)
            for ind, fit in zip(invalids, fitnesses):
                ind.fitness.values = fit

            # 精英
            if self.elitism == True:
                offspring = GPutilities.elitism(offspring, hof)
            else:
                pass
            if g % 10 == 0:
                self.clean_log(max_entry=10)  # 每10代清理一下日志

            pop[:] = offspring
            hof.update(pop)
            end_time = time.perf_counter()
            time_cost = float(end_time - start_time) * 1000.0
            best_ind = hof[0]
            if self.verbose == True:
                record = mstats.compile(pop)
                logbook.record(
                    gen=g,
                    time=time_cost,
                    num=len(pop),
                    best_ind=str(best_ind),
                    best_ind_len=len(best_ind),
                    best_ind_height=best_ind.height,
                    new=len(invalids),
                    **record,
                )
                print("------------------------------")
                print(logbook.stream)
                
                if self.best_function_dict is not None:
                    max_func, max_entry = max(self.value_log.items(),
                            key=lambda item: item[1]['count']
                        )

                    # Step 2: 如果 max_func 是新函数，初始化它的记录
                    if max_func not in self.best_function_dict:
                        # 初始化填 0 直到当前代数
                        self.best_function_dict[max_func] = [0] * g
                        print(f"新加入 best function: {max_func}")

                    # Step 3: 所有已知函数追加当前 count（即使它不是最大）
                    for func in self.best_function_dict:
                        count = self.value_log.get(func, {}).get('count', 0)
                        self.best_function_dict[func].append(count)

        self._best_ind = hof[0]
        self._best_ind_fitness = float(self._best_ind.fitness.values[0])
        self._compiled_func = gp.compile(expr=self._best_ind, pset=self.pset)
        GPutilities.SaveLogbookToPickle(logbook, self._best_ind_fitness)
        if self.tracker is True:
            for func, counts in self.best_function_dict.items():
                plt.plot(range(len(counts)), counts, label=func)
            plt.xlabel('Generation')
            plt.xticks(np.arange(0, self.gen_num, 1),fontsize=5)
            plt.ylabel('Count')
            plt.title('Count Progression of Most Subtrees in each generation')
            plt.legend(bbox_to_anchor=(1.05, 1),loc='upper left', fontsize='small')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('count_tracker.svg')
            plt.close()
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._compiled_func(*x) for x in X])