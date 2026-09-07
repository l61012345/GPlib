"""Interactive and command-line viewer for NumPy ``.npz`` archives.

Examples
--------
Open the desktop viewer::

    python npz_visualizer.py results.npz

Render selected arrays without opening a window::

    python npz_visualizer.py results.npz --arrays best_fitness mean_fitness \
        --x generations --kind line --save training.png
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.figure import Figure


AUTO_X = "<自动>"
INDEX_X = "<索引>"
MAX_LINE_POINTS = 50_000
MAX_HIST_POINTS = 500_000
MAX_IMAGE_SIDE = 2_000


def _configure_matplotlib_font() -> None:
    """Prefer an installed CJK font while retaining a portable fallback."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = (
        "Microsoft YaHei",
        "Noto Sans SC",
        "SimHei",
        "Microsoft JhengHei",
    )
    for family in preferred:
        if family in available:
            rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            break
    rcParams["axes.unicode_minus"] = False


_configure_matplotlib_font()


class VisualizerError(ValueError):
    """An error that can be shown directly to the user."""


def load_npz(path: str | os.PathLike[str], *, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    """Load and materialise all arrays so the underlying zip file can close."""
    source = Path(path).expanduser()
    if source.suffix.lower() != ".npz":
        raise VisualizerError(f"请选择 .npz 文件：{source}")
    if not source.is_file():
        raise VisualizerError(f"文件不存在：{source}")

    try:
        with np.load(source, allow_pickle=allow_pickle) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise VisualizerError(
                "文件包含对象数组。仅在信任文件来源时启用“允许对象数组”或使用 --allow-pickle。"
            ) from exc
        raise VisualizerError(f"无法读取 NPZ：{exc}") from exc
    except (OSError, EOFError) as exc:
        raise VisualizerError(f"无法读取 NPZ：{exc}") from exc


def parse_slice(spec: str, ndim: int) -> tuple[int | slice, ...]:
    """Parse a safe, small subset of NumPy indexing (integers and slices)."""
    text = spec.strip()
    if not text:
        return (slice(None),) * ndim

    parts = [part.strip() for part in text.split(",")]
    if len(parts) > ndim:
        raise VisualizerError(f"切片有 {len(parts)} 个维度，但数组只有 {ndim} 维")

    result: list[int | slice] = []
    for part in parts:
        if part in {"", ":"}:
            result.append(slice(None))
            continue
        if ":" not in part:
            try:
                result.append(int(part))
            except ValueError as exc:
                raise VisualizerError(f"无效索引：{part!r}") from exc
            continue

        fields = part.split(":")
        if len(fields) > 3:
            raise VisualizerError(f"无效切片：{part!r}")
        values: list[int | None] = []
        for field in fields:
            try:
                values.append(None if field == "" else int(field))
            except ValueError as exc:
                raise VisualizerError(f"无效切片：{part!r}") from exc
        if len(values) == 2:
            values.append(None)
        if values[2] == 0:
            raise VisualizerError("切片步长不能为 0")
        result.append(slice(values[0], values[1], values[2]))

    result.extend([slice(None)] * (ndim - len(result)))
    return tuple(result)


def apply_slice(array: np.ndarray, spec: str) -> np.ndarray:
    try:
        return np.asarray(array[parse_slice(spec, array.ndim)])
    except IndexError as exc:
        raise VisualizerError(f"切片超出数组范围：{exc}") from exc


def _is_numeric(array: np.ndarray) -> bool:
    return bool(
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    )


def _format_number(value: object) -> str:
    if isinstance(value, (np.complexfloating, complex)):
        return f"{value.real:.6g}{value.imag:+.6g}j"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}"
    return str(value)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def summarize_array(array: np.ndarray) -> dict[str, str]:
    """Return display-friendly metadata and finite numeric statistics."""
    summary = {
        "shape": str(array.shape),
        "dtype": str(array.dtype),
        "size": f"{array.size:,}",
        "bytes": _format_bytes(array.nbytes),
        "min": "—",
        "max": "—",
        "mean": "—",
    }
    if not array.size or not _is_numeric(array):
        return summary

    if np.issubdtype(array.dtype, np.complexfloating):
        finite = array[np.isfinite(array)]
        if finite.size:
            magnitudes = np.abs(finite)
            summary["min"] = _format_number(np.min(magnitudes)) + " |x|"
            summary["max"] = _format_number(np.max(magnitudes)) + " |x|"
            summary["mean"] = _format_number(np.mean(finite))
        return summary

    try:
        finite = array[np.isfinite(array)]
    except TypeError:
        return summary
    if finite.size:
        summary["min"] = _format_number(np.min(finite))
        summary["max"] = _format_number(np.max(finite))
        summary["mean"] = _format_number(np.mean(finite))
    return summary


def archive_rows(arrays: Mapping[str, np.ndarray]) -> list[tuple[str, dict[str, str]]]:
    return [(name, summarize_array(array)) for name, array in arrays.items()]


def preview_text(name: str, array: np.ndarray) -> str:
    summary = summarize_array(array)
    metadata = (
        f"名称: {name}\n形状: {summary['shape']}\n类型: {summary['dtype']}\n"
        f"元素: {summary['size']}\n内存: {summary['bytes']}\n"
        f"最小值: {summary['min']}\n最大值: {summary['max']}\n平均值: {summary['mean']}\n\n"
    )
    body = np.array2string(
        array,
        threshold=2_000,
        edgeitems=12,
        max_line_width=120,
        precision=7,
        suppress_small=False,
    )
    return metadata + body


def _as_plot_values(array: np.ndarray, name: str) -> np.ndarray:
    if not _is_numeric(array):
        raise VisualizerError(f"{name!r} 不是数值数组，不能绘图")
    if np.issubdtype(array.dtype, np.complexfloating):
        return np.abs(array)
    return array.astype(float, copy=False)


def _downsample_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y.size <= MAX_LINE_POINTS:
        return x, y
    positions = np.linspace(0, y.size - 1, MAX_LINE_POINTS, dtype=np.int64)
    return x[positions], y[positions]


def _x_values(
    arrays: Mapping[str, np.ndarray],
    x_key: str,
    length: int,
    *,
    slice_spec: str = "",
    original_length: int | None = None,
) -> tuple[np.ndarray, str]:
    candidate = x_key
    if x_key == AUTO_X:
        candidate = "generations" if "generations" in arrays else INDEX_X
    if candidate == INDEX_X:
        x = np.arange(original_length if original_length is not None else length)
        if slice_spec:
            x = apply_slice(x, slice_spec)
        if x.ndim != 1 or x.size != length:
            raise VisualizerError("切片后的索引与数据长度不一致")
        return x, "索引"
    if candidate not in arrays:
        raise VisualizerError(f"找不到 X 轴数组：{candidate}")
    x = np.asarray(arrays[candidate])
    if slice_spec:
        x = apply_slice(x, slice_spec)
    if x.ndim != 1 or x.size != length or not _is_numeric(x):
        raise VisualizerError(
            f"X 轴 {candidate!r} 必须是一维数值数组，且长度应为 {length}"
        )
    if np.issubdtype(x.dtype, np.complexfloating):
        raise VisualizerError("X 轴不能使用复数数组")
    return x.astype(float, copy=False), candidate


def _style_axis(axis) -> None:
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.set_axisbelow(True)


def _render_lines(
    figure: Figure,
    arrays: Mapping[str, np.ndarray],
    selected: list[tuple[str, np.ndarray]],
    x_key: str,
    slice_spec: str,
    *,
    scatter: bool = False,
) -> None:
    axis = figure.subplots()
    for name, raw in selected:
        if raw.ndim != 1:
            raise VisualizerError(f"{name!r} 切片后是 {raw.ndim} 维；折线/散点图需要一维数组")
        y = _as_plot_values(raw, name)
        x_slice = slice_spec if arrays[name].ndim == 1 else ""
        x, x_label = _x_values(
            arrays,
            x_key,
            y.size,
            slice_spec=x_slice,
            original_length=arrays[name].size,
        )
        x, y = _downsample_xy(x, y)
        label = f"|{name}|" if np.issubdtype(raw.dtype, np.complexfloating) else name
        if scatter:
            axis.scatter(x, y, s=13, alpha=0.75, label=label)
        else:
            axis.plot(x, y, linewidth=1.5, label=label)
    axis.set_title("散点图" if scatter else "序列曲线")
    axis.set_xlabel(x_label)
    axis.set_ylabel("值")
    if len(selected) > 1:
        axis.legend(loc="best")
    _style_axis(axis)


def _render_histogram(figure: Figure, selected: list[tuple[str, np.ndarray]]) -> None:
    axis = figure.subplots()
    for name, raw in selected:
        values = _as_plot_values(raw, name).ravel()
        values = values[np.isfinite(values)]
        if values.size > MAX_HIST_POINTS:
            positions = np.linspace(0, values.size - 1, MAX_HIST_POINTS, dtype=np.int64)
            values = values[positions]
        if not values.size:
            raise VisualizerError(f"{name!r} 没有可绘制的有限数值")
        axis.hist(values, bins=50, alpha=0.55, label=name)
    axis.set_title("数值分布")
    axis.set_xlabel("值")
    axis.set_ylabel("频数")
    if len(selected) > 1:
        axis.legend(loc="best")
    _style_axis(axis)


def _image_values(raw: np.ndarray, name: str) -> np.ndarray:
    if raw.ndim != 2:
        raise VisualizerError(f"{name!r} 切片后是 {raw.ndim} 维；热图需要二维数组")
    image = _as_plot_values(raw, name)
    row_step = max(1, math.ceil(image.shape[0] / MAX_IMAGE_SIDE))
    col_step = max(1, math.ceil(image.shape[1] / MAX_IMAGE_SIDE))
    return image[::row_step, ::col_step]


def _render_heatmaps(figure: Figure, selected: list[tuple[str, np.ndarray]]) -> None:
    count = len(selected)
    columns = min(2, count)
    rows = math.ceil(count / columns)
    axes = np.atleast_1d(figure.subplots(rows, columns, squeeze=False)).ravel()
    for axis, (name, raw) in zip(axes, selected):
        image = _image_values(raw, name)
        plotted = axis.imshow(image, aspect="auto", interpolation="nearest", origin="upper")
        axis.set_title(name)
        axis.set_xlabel("列")
        axis.set_ylabel("行")
        figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes[count:]:
        axis.remove()


def _render_auto_facets(
    figure: Figure,
    arrays: Mapping[str, np.ndarray],
    selected: list[tuple[str, np.ndarray]],
    x_key: str,
    slice_spec: str,
) -> None:
    count = len(selected)
    columns = min(2, count)
    rows = math.ceil(count / columns)
    axes = np.atleast_1d(figure.subplots(rows, columns, squeeze=False)).ravel()
    for axis, (name, raw) in zip(axes, selected):
        if raw.ndim == 0:
            axis.text(0.5, 0.5, _format_number(raw.item()), ha="center", va="center")
            axis.set_title(name)
            axis.set_axis_off()
        elif raw.ndim == 1:
            y = _as_plot_values(raw, name)
            x_slice = slice_spec if arrays[name].ndim == 1 else ""
            x, label = _x_values(
                arrays,
                x_key,
                y.size,
                slice_spec=x_slice,
                original_length=arrays[name].size,
            )
            x, y = _downsample_xy(x, y)
            axis.plot(x, y, linewidth=1.5)
            axis.set_title(name)
            axis.set_xlabel(label)
            axis.set_ylabel("值")
            _style_axis(axis)
        elif raw.ndim == 2:
            image = _image_values(raw, name)
            plotted = axis.imshow(image, aspect="auto", interpolation="nearest", origin="upper")
            axis.set_title(name)
            axis.set_xlabel("列")
            axis.set_ylabel("行")
            figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
        else:
            raise VisualizerError(
                f"{name!r} 切片后仍有 {raw.ndim} 维；请用切片将它降到二维或一维"
            )
    for axis in axes[count:]:
        axis.remove()


def render_figure(
    arrays: Mapping[str, np.ndarray],
    names: Iterable[str],
    *,
    kind: str = "auto",
    x_key: str = AUTO_X,
    slice_spec: str = "",
    figure: Figure | None = None,
) -> Figure:
    """Render selected arrays into a Matplotlib figure."""
    chosen = list(dict.fromkeys(names))
    if not chosen:
        raise VisualizerError("请至少选择一个数组")
    missing = [name for name in chosen if name not in arrays]
    if missing:
        raise VisualizerError("找不到数组：" + ", ".join(missing))
    if len(chosen) > 16:
        raise VisualizerError("一次最多绘制 16 个数组")

    selected = [(name, apply_slice(arrays[name], slice_spec)) for name in chosen]
    figure = figure or Figure(figsize=(9, 6), dpi=100)
    figure.clear()

    if kind == "auto":
        dimensions = {array.ndim for _, array in selected}
        if dimensions == {1}:
            _render_lines(figure, arrays, selected, x_key, slice_spec)
        elif dimensions == {2}:
            _render_heatmaps(figure, selected)
        else:
            _render_auto_facets(figure, arrays, selected, x_key, slice_spec)
    elif kind == "line":
        _render_lines(figure, arrays, selected, x_key, slice_spec)
    elif kind == "scatter":
        _render_lines(figure, arrays, selected, x_key, slice_spec, scatter=True)
    elif kind == "hist":
        _render_histogram(figure, selected)
    elif kind == "heatmap":
        _render_heatmaps(figure, selected)
    else:
        raise VisualizerError(f"未知图表类型：{kind}")

    figure.tight_layout()
    return figure


def print_archive(arrays: Mapping[str, np.ndarray]) -> None:
    headings = ("名称", "形状", "类型", "元素", "内存", "最小值", "最大值", "平均值")
    rows = []
    for name, summary in archive_rows(arrays):
        rows.append(
            (
                name,
                summary["shape"],
                summary["dtype"],
                summary["size"],
                summary["bytes"],
                summary["min"],
                summary["max"],
                summary["mean"],
            )
        )
    widths = [len(title) for title in headings]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print("  ".join(title.ljust(width) for title, width in zip(headings, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


class NpzVisualizerApp:
    """Tk desktop application; imports Tk only when the GUI is requested."""

    KIND_LABELS = {
        "自动": "auto",
        "折线图": "line",
        "散点图": "scatter",
        "直方图": "hist",
        "热图": "heatmap",
    }

    def __init__(self, root, initial_path: str | None = None, *, allow_pickle: bool = False):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("NPZ 可视化工具")
        self.root.geometry("1180x760")
        self.root.minsize(820, 560)
        self.path: Path | None = None
        self.arrays: dict[str, np.ndarray] = {}
        self._render_job = None

        self.allow_pickle = tk.BooleanVar(value=allow_pickle)
        self.path_text = tk.StringVar(value="尚未打开文件")
        self.kind_text = tk.StringVar(value="自动")
        self.x_text = tk.StringVar(value=AUTO_X)
        self.slice_text = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="打开一个 .npz 文件开始")

        self._build_menu()

        top = ttk.Frame(root, padding=(10, 8))
        top.pack(fill=tk.X)
        ttk.Button(top, text="打开…", command=self.open_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="重新加载", command=self.reload).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(top, textvariable=self.path_text).pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        pane = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(pane, padding=(0, 0, 8, 0))
        right = ttk.Frame(pane)
        pane.add(left, weight=1)
        pane.add(right, weight=4)

        ttk.Label(left, text="数组（可多选）").pack(anchor=tk.W, pady=(0, 5))
        columns = ("shape", "dtype", "size")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="名称")
        self.tree.heading("shape", text="形状")
        self.tree.heading("dtype", text="类型")
        self.tree.heading("size", text="元素")
        self.tree.column("#0", width=145, minwidth=90)
        self.tree.column("shape", width=100, minwidth=70)
        self.tree.column("dtype", width=75, minwidth=55)
        self.tree.column("size", width=80, minwidth=55, anchor=tk.E)
        tree_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)

        controls = ttk.Frame(right)
        controls.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(controls, text="图表").grid(row=0, column=0, sticky=tk.W)
        kind = ttk.Combobox(
            controls,
            textvariable=self.kind_text,
            values=list(self.KIND_LABELS),
            state="readonly",
            width=9,
        )
        kind.grid(row=1, column=0, sticky=tk.EW, padx=(0, 6))
        kind.bind("<<ComboboxSelected>>", self._selection_changed)

        ttk.Label(controls, text="X 轴").grid(row=0, column=1, sticky=tk.W)
        self.x_combo = ttk.Combobox(controls, textvariable=self.x_text, state="readonly", width=16)
        self.x_combo.grid(row=1, column=1, sticky=tk.EW, padx=(0, 6))
        self.x_combo.bind("<<ComboboxSelected>>", self._selection_changed)

        ttk.Label(controls, text="切片（如 0, : 或 :, 10:50）").grid(row=0, column=2, sticky=tk.W)
        slice_entry = ttk.Entry(controls, textvariable=self.slice_text, width=24)
        slice_entry.grid(row=1, column=2, sticky=tk.EW, padx=(0, 6))
        slice_entry.bind("<Return>", self.render)
        ttk.Button(controls, text="绘制", command=self.render).grid(row=1, column=3, padx=(0, 6))
        ttk.Button(controls, text="导出图片…", command=self.export_dialog).grid(row=1, column=4)
        controls.columnconfigure(2, weight=1)

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)
        plot_tab = ttk.Frame(notebook)
        preview_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="图表")
        notebook.add(preview_tab, text="数据预览")

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_tab)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.preview = tk.Text(preview_tab, wrap="none", font="TkFixedFont")
        preview_y = ttk.Scrollbar(preview_tab, orient=tk.VERTICAL, command=self.preview.yview)
        preview_x = ttk.Scrollbar(preview_tab, orient=tk.HORIZONTAL, command=self.preview.xview)
        self.preview.configure(yscrollcommand=preview_y.set, xscrollcommand=preview_x.set)
        preview_y.pack(side=tk.RIGHT, fill=tk.Y)
        preview_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview.configure(state="disabled")

        ttk.Label(root, textvariable=self.status_text, anchor=tk.W, padding=(10, 3)).pack(fill=tk.X)

        root.bind("<Control-o>", lambda _event: self.open_dialog())
        root.bind("<Control-s>", lambda _event: self.export_dialog())
        root.bind("<F5>", lambda _event: self.reload())
        if initial_path:
            self.open_file(initial_path)

    def _build_menu(self) -> None:
        tk = self.tk
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="打开…", accelerator="Ctrl+O", command=self.open_dialog)
        file_menu.add_command(label="重新加载", accelerator="F5", command=self.reload)
        file_menu.add_command(label="导出图片…", accelerator="Ctrl+S", command=self.export_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.destroy)
        menu.add_cascade(label="文件", menu=file_menu)

        options = tk.Menu(menu, tearoff=False)
        options.add_checkbutton(
            label="允许对象数组（仅限可信文件）",
            variable=self.allow_pickle,
        )
        menu.add_cascade(label="选项", menu=options)
        self.root.configure(menu=menu)

    def open_dialog(self) -> None:
        from tkinter import filedialog

        filename = filedialog.askopenfilename(
            title="打开 NPZ 文件",
            filetypes=(("NumPy 压缩文件", "*.npz"), ("所有文件", "*.*")),
        )
        if filename:
            self.open_file(filename)

    def open_file(self, path: str | os.PathLike[str]) -> None:
        from tkinter import messagebox

        try:
            arrays = load_npz(path, allow_pickle=self.allow_pickle.get())
        except VisualizerError as exc:
            messagebox.showerror("打开失败", str(exc))
            self.status_text.set(str(exc))
            return

        self.path = Path(path).resolve()
        self.arrays = arrays
        self.path_text.set(str(self.path))
        self.tree.delete(*self.tree.get_children())
        for name, summary in archive_rows(arrays):
            self.tree.insert(
                "",
                "end",
                iid=name,
                text=name,
                values=(summary["shape"], summary["dtype"], summary["size"]),
            )

        one_dimensional = [
            name for name, array in arrays.items() if array.ndim == 1 and _is_numeric(array)
        ]
        self.x_combo.configure(values=[AUTO_X, INDEX_X, *one_dimensional])
        self.x_text.set(AUTO_X)

        preferred = [
            name
            for name in one_dimensional
            if name != "generations"
            and ("generations" not in arrays or arrays[name].size == arrays["generations"].size)
        ][:4]
        if not preferred and arrays:
            preferred = [next(iter(arrays))]
        for name in preferred:
            self.tree.selection_add(name)
        if preferred:
            self.tree.focus(preferred[0])
            self.tree.see(preferred[0])
        self.status_text.set(f"已加载 {len(arrays)} 个数组 — {self.path.name}")
        self._update_preview()
        self.render()

    def reload(self) -> None:
        if self.path:
            self.open_file(self.path)

    def selected_names(self) -> list[str]:
        return list(self.tree.selection())

    def _selection_changed(self, _event=None) -> None:
        self._update_preview()
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(120, self.render)

    def _update_preview(self) -> None:
        names = self.selected_names()
        if not names:
            text = "请选择一个数组。"
        elif len(names) == 1:
            text = preview_text(names[0], self.arrays[names[0]])
        else:
            blocks = []
            for name in names:
                summary = summarize_array(self.arrays[name])
                blocks.append(
                    f"{name}: shape={summary['shape']}, dtype={summary['dtype']}, "
                    f"min={summary['min']}, max={summary['max']}, mean={summary['mean']}"
                )
            text = "\n".join(blocks)
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def render(self, _event=None) -> bool:
        self._render_job = None
        names = self.selected_names()
        if not names:
            return False
        try:
            render_figure(
                self.arrays,
                names,
                kind=self.KIND_LABELS[self.kind_text.get()],
                x_key=self.x_text.get(),
                slice_spec=self.slice_text.get(),
                figure=self.figure,
            )
            self.canvas.draw_idle()
            self.status_text.set(f"已绘制 {len(names)} 个数组")
            return True
        except (VisualizerError, ValueError, TypeError) as exc:
            self.figure.clear()
            axis = self.figure.subplots()
            axis.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True)
            axis.set_axis_off()
            self.canvas.draw_idle()
            self.status_text.set(str(exc))
            return False

    def export_dialog(self) -> None:
        from tkinter import filedialog, messagebox

        if not self.arrays or not self.selected_names():
            messagebox.showinfo("没有图表", "请先打开文件并选择要绘制的数组。")
            return
        filename = filedialog.asksaveasfilename(
            title="导出图表",
            defaultextension=".png",
            filetypes=(
                ("PNG 图片", "*.png"),
                ("PDF 文档", "*.pdf"),
                ("SVG 矢量图", "*.svg"),
            ),
        )
        if not filename:
            return
        try:
            if not self.render():
                messagebox.showerror("导出失败", self.status_text.get())
                return
            self.figure.savefig(filename, dpi=180, bbox_inches="tight")
        except (OSError, ValueError) as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.status_text.set(f"已导出：{filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="浏览和绘制 NumPy .npz 文件")
    parser.add_argument("path", nargs="?", help="要打开的 .npz 文件")
    parser.add_argument("--list", action="store_true", help="列出数组信息后退出")
    parser.add_argument("--arrays", nargs="+", metavar="NAME", help="要绘制的数组名称")
    parser.add_argument(
        "--kind",
        choices=("auto", "line", "scatter", "hist", "heatmap"),
        default="auto",
        help="图表类型（默认：auto）",
    )
    parser.add_argument("--x", default=AUTO_X, help="X 轴数组名称；使用 index 表示元素索引")
    parser.add_argument("--slice", dest="slice_spec", default="", help="安全 NumPy 切片，如 '0, :' 或 ':, 10:50'")
    parser.add_argument("--save", metavar="IMAGE", help="直接保存图片，不打开窗口")
    parser.add_argument("--dpi", type=int, default=180, help="导出图片 DPI（默认：180）")
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="允许读取对象数组；只应对可信文件使用",
    )
    return parser


def _normalise_x_argument(value: str) -> str:
    if value.lower() in {"auto", AUTO_X.lower()}:
        return AUTO_X
    if value.lower() in {"index", INDEX_X.lower()}:
        return INDEX_X
    return value


def run_cli(args: argparse.Namespace) -> int:
    if not args.path:
        raise VisualizerError("使用 --list 或 --save 时必须提供 NPZ 文件路径")
    arrays = load_npz(args.path, allow_pickle=args.allow_pickle)
    if args.list:
        print_archive(arrays)
    if args.save:
        names = args.arrays or [
            name for name, array in arrays.items() if name != "generations" and _is_numeric(array)
        ][:4]
        if not names:
            names = [name for name, array in arrays.items() if _is_numeric(array)][:4]
        figure = render_figure(
            arrays,
            names,
            kind=args.kind,
            x_key=_normalise_x_argument(args.x),
            slice_spec=args.slice_spec,
        )
        output = Path(args.save).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
        print(f"已保存：{output.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Some Windows shells still expose a legacy code page to redirected Python
    # processes. UTF-8 keeps the Chinese CLI help and status output portable.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.list or args.save:
            return run_cli(args)

        import tkinter as tk

        root = tk.Tk()
        NpzVisualizerApp(root, args.path, allow_pickle=args.allow_pickle)
        root.mainloop()
        return 0
    except VisualizerError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
