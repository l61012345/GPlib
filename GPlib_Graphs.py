import os
import pickle

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
            matplotlib.use("Agg", force=True)  # 后台绘图，不弹窗

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

    def save_tracker_pkl(self, path=None):
        import pickle
        import os

        if path is None:
            path = self.filename

        if not path.endswith(".pkl"):
            path = f"{path}.pkl"

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "generations": self.generations,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "mean_size": self.mean_size,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_tracker_pkl(self, path):
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.generations = data["generations"]
        self.best_fitness = data["best_fitness"]
        self.mean_fitness = data["mean_fitness"]
        self.mean_size = data["mean_size"]

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
        self.save_tracker_pkl(self.filename)

        # live display
        if self.LiveDisplay:
            self.plt.pause(0.01)


class AdaptiveGraphTracker:
    """
    通用多子图追踪器

    参数
    ----
    tracked_layout : list
        控制子图布局。
        例如：
            [
                ["best_fitness", "mean_fitness"],
                "mean_size",
                ["tie_score_mean", "tie_score_var"],
                "potential_count",
            ]

        含义：
        - 第1个子图画 best_fitness 和 mean_fitness
        - 第2个子图画 mean_size
        - 第3个子图画 tie_score_mean 和 tie_score_var
        - 第4个子图画 potential_count

    LiveDisplay : bool
        是否实时显示
    filename : str
        输出文件名前缀（不带后缀）
    dpi : int
        保存图片 dpi
    format : str
        保存图片格式，如 "tiff" / "png"
    figsize : tuple | None
        图尺寸，None 时自动按子图数量调整
    style_map : dict | None
        每条曲线的样式映射，例如：
            {
                "best_fitness": {"c": "tab:blue", "marker": "o"},
                "mean_fitness": {"c": "tab:orange", "marker": "x"},
            }
    title_map : dict | None
        子图标题映射。
        单线可用键："mean_size"
        多线可用键：("best_fitness", "mean_fitness")
    ylabel_map : dict | None
        子图 y 轴标题映射，规则同 title_map
    fmt_map : dict | None
        每条曲线数值标签格式，例如：
            {"potential_count": "{:.0f}", "mean_size": "{:.1f}"}
    step_map : dict | None
        每条曲线的标注间隔。
        若未提供，则自动按点数估计
    """

    def __init__(
        self,
        tracked_layout,
        *,
        LiveDisplay=True,
        filename="adaptive_training_curve",
        dpi=550,
        format="tiff",
        figsize=None,
        style_map=None,
        title_map=None,
        name_map=None,
        ylabel_map=None,
        fmt_map=None,
        step_map=None,
    ):
        self.LiveDisplay = LiveDisplay
        self.filename = filename
        self.dpi = dpi
        self.format = format

        if not LiveDisplay:
            matplotlib.use("Agg",force=True)  # 后台绘图，不弹窗

        import matplotlib.pyplot as plt
        self.plt = plt

        if self.LiveDisplay:
            self.plt.ion()

        # 统一把每个 panel 处理成 list[str]
        self.tracked_layout = [
            list(item) if isinstance(item, (list, tuple)) else [item]
            for item in tracked_layout
        ]

        # 记录代数
        self.generations = []

        # 为每条曲线准备独立存储
        self.series = {}
        for panel in self.tracked_layout:
            for name in panel:
                self.series[name] = []

        self.style_map = style_map or {}
        self.title_map = title_map or {}
        self.ylabel_map = ylabel_map or {}
        self.fmt_map = fmt_map or {}
        self.step_map = step_map or {}
        self.name_map = name_map or {}

        n_subplots = len(self.tracked_layout)
        if figsize is None:
            figsize = (7, max(4, 3.6 * n_subplots))

        self.fig, axes = self.plt.subplots(n_subplots, 1, figsize=figsize)
        if n_subplots == 1:
            axes = [axes]
        self.axes = list(axes)

    # =========================================================
    # 数据更新
    # =========================================================
    def update(self, gen, **tracked_values):
        """
        更新一代的数据

        示例
        ----
        tracker.update(
            gen,
            best_fitness=best_fit,
            mean_fitness=mean_fit,
            mean_size=mean_size,
            tie_score_mean=tie_mean,
            tie_score_var=tie_var,
            potential_count=potential_count,
        )
        """
        self.generations.append(gen)

        for name in self.series:
            value = tracked_values.get(name, np.nan)
            self.series[name].append(value)

    def update_from_dict(self, gen, stats_dict):
        """
        用 dict 更新，便于主循环直接传统计量字典
        """
        self.update(gen, **stats_dict)

    # =========================================================
    # 保存 / 读取
    # =========================================================
    def save_tracker_pkl(self, path=None):
        if path is None:
            path = self.filename
        if not path.endswith(".pkl"):
            path = f"{path}.pkl"

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "generations": self.generations,
            "tracked_layout": self.tracked_layout,
            "series": self.series,
            "filename": self.filename,
            "dpi": self.dpi,
            "format": self.format,
            "style_map": self.style_map,
            "title_map": self.title_map,
            "ylabel_map": self.ylabel_map,
            "fmt_map": self.fmt_map,
            "step_map": self.step_map,
            "name_map": self.name_map,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_tracker_pkl(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.generations = data["generations"]
        self.tracked_layout = data["tracked_layout"]
        self.series = data["series"]

        self.filename = data.get("filename", self.filename)
        self.dpi = data.get("dpi", self.dpi)
        self.format = data.get("format", self.format)
        self.style_map = data.get("style_map", self.style_map)
        self.title_map = data.get("title_map", self.title_map)
        self.ylabel_map = data.get("ylabel_map", self.ylabel_map)
        self.fmt_map = data.get("fmt_map", self.fmt_map)
        self.step_map = data.get("step_map", self.step_map)
        self.name_map = data.get("name_map", self.name_map)

    # =========================================================
    # 样式 / 标题辅助
    # =========================================================
    def _default_style_for_index(self, idx):
        defaults = [
            dict(marker="o", markersize=1, c="tab:blue"),
            dict(marker="x", markersize=1, c="tab:orange"),
            dict(marker="s", markersize=1, c="tab:green"),
            dict(marker="^", markersize=1, c="tab:red"),
            dict(marker="d", markersize=1, c="tab:purple"),
            dict(marker="v", markersize=1, c="tab:brown"),
        ]
        return defaults[idx % len(defaults)].copy()

    def _panel_key(self, names):
        return tuple(names) if len(names) > 1 else names[0]

    def _panel_title(self, names):
        key = self._panel_key(names)
        if key in self.title_map:
            return self.title_map[key]

        if len(names) == 1:
            return self.name_map.get(names[0], names[0])
        return " / ".join(self.name_map.get(n, n) for n in names)

    def _panel_ylabel(self, names):
        key = self._panel_key(names)
        if key in self.ylabel_map:
            return self.ylabel_map[key]

        if len(names) == 1:
            return self.name_map.get(names[0], names[0])
        return "Value"

    def _label_fmt(self, name):
        return self.fmt_map.get(name, "{:.4f}")

    def _label_step(self, name, n_points):
        if name in self.step_map:
            return max(1, int(self.step_map[name]))
        return max(1, int(np.ceil(n_points / 10)))

    # =========================================================
    # 标注：采用 size 的方案
    # =========================================================
    def annotate_sparse_points(
        self,
        ax,
        xs,
        ys,
        *,
        fmt="{:.4f}",
        fontsize=8,
        step=None,
        keep_last=True,
        xytext=(0, 5),
    ):
        """
        类似原先 size 的标注方式：
        - 每隔 step 个点标一次
        - 最后一个点强制标注
        """
        n = len(xs)
        if n == 0:
            return

        if step is None:
            step = max(1, n // 12)

        for i, (x, y) in enumerate(zip(xs, ys)):
            if np.isnan(y):
                continue

            if i % step != 0 and not (keep_last and i == n - 1):
                continue

            ax.annotate(
                fmt.format(y),
                (x, y),
                textcoords="offset points",
                xytext=xytext,
                ha="center",
                fontsize=fontsize,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.8,
                ),
            )

    # =========================================================
    # 绘图
    # =========================================================
    def plot(self):
        n_points = len(self.generations)
        fontsize = max(6, 8 - n_points // 10)

        for ax in self.axes:
            ax.clear()

        for panel_idx, (ax, names) in enumerate(zip(self.axes, self.tracked_layout)):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.tick_params(axis="y", pad=6)

            # 先画线
            for line_idx, name in enumerate(names):
                xs = self.generations
                ys = self.series[name]

                style = self._default_style_for_index(line_idx)
                style.update(self.style_map.get(name, {}))
                label = self.name_map.get(name, name)
                ax.plot(xs, ys, label=label, **style)

            # 再做稀疏标注
            for line_idx, name in enumerate(names):
                xs = self.generations
                ys = self.series[name]

                fmt = self._label_fmt(name)
                step = self._label_step(name, len(xs))

                # 多线时稍微错开，避免标签完全压在一起
                xytext = (0, 5 + 8 * line_idx)

                self.annotate_sparse_points(
                    ax,
                    xs,
                    ys,
                    fmt=fmt,
                    fontsize=fontsize,
                    step=step,
                    keep_last=True,
                    xytext=xytext,
                )

            ax.legend(fontsize=fontsize, ncol=2)
            ax.set_title(self._panel_title(names))
            ax.set_xlabel("Generation")
            ax.set_ylabel(self._panel_ylabel(names))

        self.fig.tight_layout(rect=[0.12, 0.0, 0.95, 1.0])

        # 保存图片
        self.fig.savefig(
            f"{self.filename}.{self.format}",
            dpi=self.dpi,
            format=self.format,
        )

        # 保存数据
        self.save_tracker_pkl(self.filename)

        if self.LiveDisplay:
            self.plt.pause(0.01)