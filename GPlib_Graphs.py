import numpy as np
import matplotlib

class GraphTracker:

    def __init__(self, LiveDisplay=True, filename="gp_training_curve",dpi=550,format="tiff"):
        self.LiveDisplay = LiveDisplay
        self.filename = filename
        self.dpi = dpi
        self.format = format

        # === backend 控制 ===
        if not LiveDisplay:
            matplotlib.use("Agg")  # 后台绘图，不弹窗

        import matplotlib.pyplot as plt
        self.plt = plt

        # === 数据记录 ===
        self.generations = []
        self.best_fitness = []
        self.mean_fitness = []
        self.mean_size = []

        # === 图像初始化 ===
        if self.LiveDisplay:
            self.plt.ion()

        self.fig, (self.ax1, self.ax2) = self.plt.subplots(2, 1, figsize=(7, 8))

        self.ax1.set_title("Fitness")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Fitness")

        self.ax2.set_title("Average Tree Size")
        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Nodes")

    def update(self, gen, population):

        fits = [ind.fitness.values[0] for ind in population]
        sizes = [len(ind) for ind in population]

        self.generations.append(gen)
        self.best_fitness.append(np.max(fits))
        self.mean_fitness.append(np.mean(fits))
        self.mean_size.append(np.mean(sizes))

    def annotate_changed_points(self, ax, xs, ys, *, 
                                fmt="{:.4f}", xytext=(0, 5), fontsize=8, ha="center",atol=1e-12,show_first=True
                                ):
        """
        只给“相对前一个值发生变化”的点添加标注。

        参数
        ----
        ax : matplotlib.axes.Axes
            目标坐标轴
        xs, ys : iterable
            横纵坐标序列
        fmt : str
            标注格式
        xytext : tuple
            annotate 的偏移
        fontsize : int
            字体大小
        ha : str
            水平对齐
        atol : float
            浮点比较容差
        show_first : bool
            是否显示第一个点的标注
        """
        prev = None

        for i, (x, y) in enumerate(zip(xs, ys)):
            should_show = False

            if i == 0:
                should_show = show_first
            else:
                if not np.isclose(y, prev, atol=atol):
                    should_show = True

            if should_show:
                ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=xytext, ha=ha, fontsize=fontsize)

            prev = y

    def plot(self):
        n_points = len(self.generations)
        fontsize = max(2, 8 - n_points // 10)
        self.ax1.clear()
        self.ax2.clear()

        # === fitness 曲线 ===
        self.ax1.plot(self.generations, self.best_fitness, label="Best",marker='o',markersize=1)
        self.ax1.plot(self.generations, self.mean_fitness, label="Mean",marker='x',markersize=1)

        # 动态标注数值
        self.annotate_changed_points(self.ax1,self.generations,self.best_fitness,
                fmt="{:.4f}",xytext=(0, 5), fontsize=fontsize, atol=1e-12, show_first=True)
        self.annotate_changed_points(self.ax1,self.generations,self.mean_fitness,
                fmt="{:.4f}",xytext=(0, 5), fontsize=fontsize, atol=1e-12, show_first=True)

        self.ax1.legend()

        self.ax1.set_title("Fitness")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Fitness")

        # === size 曲线 ===
        self.ax2.plot(self.generations, self.mean_size, marker='s',markersize=1)
        for x, y in zip(self.generations, self.mean_size):
            self.ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=fontsize)
        self.ax2.set_title("Average Tree Size")
        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Nodes")

        self.fig.tight_layout()

        # === 保存图像 ===
        self.fig.savefig(self.filename, dpi=self.dpi, format=self.format)

        # === Live display ===
        if self.LiveDisplay:
            self.plt.pause(0.01)