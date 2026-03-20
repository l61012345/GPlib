import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator


class GraphTracker:

    def __init__(
        self, LiveDisplay=True, filename="gp_training_curve", dpi=550, format="tiff"
    ):
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

    def save_fig_pkl(self, path=None):
        """
        保存当前 matplotlib figure 为 .pkl（可重新加载并继续编辑）
        """
        import pickle
        import os

        if path is None:
            path = self.filename

        if not path.endswith(".pkl"):
            path = f"{path}.pkl"

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.fig, f)


    def annotate_changed_points(self,ax,xs,ys,*,
        fmt="{:.4f}",
        fontsize=8,
        atol=1e-12,
        show_first=True,
        side="left",
        line_alpha=0.7,
        line_style=":",
        line_color="gray",
        axis_text_offset=6,
        min_label_gap_px=8,
        min_delta_to_label=None,
        draw_line=True,
        keep_last_point_label=True,
    ):
        import numpy as np

        if not xs or not ys:
            return

        ax.figure.canvas.draw()
        trans = ax.get_yaxis_transform()

        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")

        x_text = 0.0 if side == "left" else 1.0
        x_offset = -abs(axis_text_offset) if side == "left" else abs(axis_text_offset)
        ha_axis = "right" if side == "left" else "left"

        prev_y = None
        placed_y_pixels = []

        # ---------- 1) 轴侧稀疏标签 ----------
        for i, y in enumerate(ys):
            changed = show_first if i == 0 else (
                not np.isclose(y, prev_y, atol=atol)
                if min_delta_to_label is None
                else abs(y - prev_y) >= min_delta_to_label
            )

            if not changed:
                prev_y = y
                continue

            _, y_disp = ax.transData.transform((0, y))
            if any(abs(y_disp - py) < min_label_gap_px for py in placed_y_pixels):
                prev_y = y
                continue

            if draw_line:
                ax.axhline(
                    y=y, linestyle=line_style, linewidth=0.7,
                    alpha=line_alpha, zorder=0, c=line_color
                )

            ax.annotate(
                fmt.format(y),
                xy=(x_text, y),
                xycoords=trans,
                textcoords="offset points",
                xytext=(x_offset, 0),
                ha=ha_axis,
                va="center",
                fontsize=fontsize,
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=5,
            )

            placed_y_pixels.append(y_disp)
            prev_y = y

        # ---------- 2) 最后一个点图内标签 ----------
        if not keep_last_point_label:
            return

        x_last, y_last = xs[-1], ys[-1]
        if not hasattr(ax, "_last_endpoint_labels"):
            ax._last_endpoint_labels = []

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)

        x_ratio = (x_last - x_min) / x_span
        y_ratio = (y_last - y_min) / y_span

        dx = -8 if x_ratio > 0.90 else 8
        ha_last = "right" if dx < 0 else "left"
        dy = -6 if y_ratio > 0.90 else 6
        va_last = "top" if dy < 0 else "bottom"

        y_close_thresh = y_span * 0.025
        if any(abs(y_last - info["y"]) < y_close_thresh for info in ax._last_endpoint_labels):
            dy = -8 if dy > 0 else 8
            va_last = "top" if dy < 0 else "bottom"
            if x_ratio <= 0.90:
                dx = -8 if dx > 0 else 8
                ha_last = "right" if dx < 0 else "left"

        ax.annotate(
            fmt.format(y_last),
            xy=(x_last, y_last),
            xycoords="data",
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha_last,
            va=va_last,
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="-", lw=0.6, alpha=0.6),
            clip_on=False,
            zorder=6,
        )

        ax._last_endpoint_labels.append({"x": x_last, "y": y_last, "dx": dx, "dy": dy})

    def plot(self):
        n_points = len(self.generations)
        fontsize = max(6, 8 - n_points // 10)

        self.ax1.clear()
        self.ax2.clear()
        if hasattr(self.ax1, "_last_endpoint_labels"):
            delattr(self.ax1, "_last_endpoint_labels")

        # =========================
        # 1) Fitness
        # =========================
        self.ax1.plot(
            self.generations,
            self.best_fitness,
            label="Best",
            marker="o",
            markersize=1,
            c="tab:blue",
        )
        self.ax1.plot(
            self.generations,
            self.mean_fitness,
            label="Mean",
            marker="x",
            markersize=1,
            c="tab:orange",
        )

        self.ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
        self.ax1.tick_params(axis="y", pad=6)
        ticks = self.ax1.get_yticks()
        labels = [f"{t:g}" for t in ticks]
        if len(labels) > 0:
            labels[-1] = "" # 去掉最后一个标签，避免和最后一个点的标签重叠
            labels[-2] = ""
        self.ax1.set_yticks(ticks)
        self.ax1.set_yticklabels(labels)

        # 左侧：Best
        self.annotate_changed_points(self.ax1, self.generations,self.best_fitness,fmt="{:.4f}",
            fontsize=fontsize,atol=1e-12,show_first=True,side="left",min_label_gap_px=10,
            min_delta_to_label=0.0015,keep_last_point_label=True,line_color="tab:blue",
        )

        # 右侧：Mean
        self.annotate_changed_points(self.ax1,self.generations,self.mean_fitness,fmt="{:.4f}",
            fontsize=fontsize,atol=1e-12,show_first=True,side="right",min_label_gap_px=10,
            min_delta_to_label=0.01,keep_last_point_label=True,line_color="tab:orange",
        )

        self.ax1.legend()
        self.ax1.set_title("Fitness")
        self.ax1.set_xlabel("Generation")
        self.ax1.set_ylabel("Fitness")

        # =========================
        # 2) Size
        # =========================
        self.ax2.plot(self.generations, self.mean_size, marker="s", markersize=1)

        step = max(1, len(self.generations) // 12)  # 控制大概显示 ~12 个标签

        for i, (x, y) in enumerate(zip(self.generations, self.mean_size)):
            if i % step != 0 and i != len(self.generations) - 1:
                continue
            self.ax2.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=fontsize,
                bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.8
    )
            )

        self.ax2.set_title("Average Tree Size")
        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Nodes")

        # 给左右标签留空间
        self.fig.subplots_adjust(left=0.18, right=0.82)

        self.fig.tight_layout()

        # 保存
        self.fig.savefig(
            f"{self.filename}.{self.format}", dpi=self.dpi, format=self.format
        )
        self.save_fig_pkl(self.filename)

        # live display
        if self.LiveDisplay:
            self.plt.pause(0.01)
