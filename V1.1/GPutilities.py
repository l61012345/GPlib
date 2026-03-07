import pickle
from datetime import datetime
from deap import tools

def printTreeList(indiv):
    # 打印一个个体的树列表
    for tree in indiv:
        print(tree)

def SaveLogbookToPickle(logbook, tobesaved_data,mark=None):
    # 保存日志文件到文件
    def save_logbook_to_pickle(logbook, filename):
        with open(filename, 'wb') as file:
            pickle.dump(logbook, file)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mark is not None:
        filename = f'logbook_{current_time}_{tobesaved_data}_{mark}.pkl'
    else:
        filename = f'logbook_{current_time}_{tobesaved_data}.pkl'
    save_logbook_to_pickle(logbook, filename)

def elitism(offspring, hof):
    # 将offspring中的最差的hofsize个个体替换为精英池中的个体
    worst = tools.selWorst(offspring, len(hof))  # 查找变异后的offspring中最差的个体
    offspring_temp = []  # 清空offspring temp
    for indiv in offspring:
        offspring_temp.append(indiv)
        if indiv in worst:  # 如果个体在worst中
            offspring_temp.remove(indiv)  # 退回刚才的添加
            worst.remove(indiv)  # worst中删除这个个体
            # ↑如此避免一次全部移除，从而保持种群大小恒定
    offspring = offspring_temp  # 将offspring_temp完全替代offspring
    offspring.extend(hof.items)  # 将精英池的种群添加到现有种群中
    return offspring