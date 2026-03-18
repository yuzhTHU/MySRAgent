# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Literal, Tuple


def get_fig(RN, CN, FW=None, FH=None, AW=None, AH=None, A_ratio=1.0, 
            LM=3, RM=3, TM=3, BM=3, HS=None, VS=None, dpi=300,
            fontsize=7, lw=0.5, gridspec=False, **kwargs):
    # 基础字体和尺寸
    plt.rcParams['font.family'] = kwargs.get('font_family', 'Arial')
    plt.rcParams['font.size'] = fontsize
    # --- 字体大小设置 (Font Sizes) ---
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize * (5/7)
    plt.rcParams['ytick.labelsize'] = fontsize * (5/7)
    plt.rcParams['legend.fontsize'] = fontsize * (5/7)
    plt.rcParams['figure.titlesize'] = fontsize
    # --- 线条宽度设置 (Line Widths) ---
    plt.rcParams['lines.linewidth'] = lw
    plt.rcParams['axes.linewidth'] = lw
    plt.rcParams['xtick.major.width'] = lw
    plt.rcParams['ytick.major.width'] = lw
    plt.rcParams['xtick.minor.width'] = lw
    plt.rcParams['ytick.minor.width'] = lw
    plt.rcParams['grid.linewidth'] = lw
    plt.rcParams['patch.linewidth'] = lw 
    # --- 导出设置 (Export Settings) ---
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    LM = LM * fontsize / 72
    RM = RM * fontsize / 72
    TM = TM * fontsize / 72
    BM = BM * fontsize / 72
    HS = HS * fontsize / 72 if HS is not None else LM
    VS = VS * fontsize / 72 if VS is not None else TM

    if AW is not None:
        AW /= 2.54
        FW = LM + CN * AW + (CN - 1) * HS + RM
        if AH is not None:
            AH /= 2.54
            FH = TM + RN * AH + (RN - 1) * VS + BM
            A_ratio = AW / AH
        else:
            AH = AW / A_ratio
            FH = TM + RN * AH + (RN - 1) * VS + BM
    elif AH is not None:
        AH /= 2.54
        AW = AH * A_ratio
        FH = TM + RN * AH + (RN - 1) * VS + BM
        FW = LM + CN * AW + (CN - 1) * HS + RM
    elif FW is not None:
        FW /= 2.54
        AW = (FW - LM - RM - (CN - 1) * HS) / CN
        if FH is not None:
            FH /= 2.54
            AH = (FH - TM - BM - (RN - 1) * VS) / RN
            A_ratio = AW / AH
        else:
            AH = AW / A_ratio
            FH = TM + RN * AH + (RN - 1) * VS + BM
    elif FH is not None:
        FH /= 2.54
        AH = (FH - TM - BM - (RN - 1) * VS) / RN
        AW = AH * A_ratio
        FW = LM + CN * AW + (CN - 1) * HS + RM
    
    figinfo = dict(RN=RN, CN=CN, FW=FW, FH=FH, AW=AW, AH=AH, 
                   LM=LM, RM=RM, TM=TM, BM=BM, HS=HS, VS=VS,
                   r_AW=AW/FW, r_AH=AH/FH, r_HS=HS/FW, r_VS=VS/FH,
                   r_LM=LM/FW, r_RM=RM/FW, r_TM=TM/FH, r_BM=BM/FH,
                   fontsize=fontsize, lw=lw,
                   top_box=(LM/FW, 1-TM/FH, 1-(LM+RM)/FW, TM/FH), 
                   bottom_box=(LM/FW, 0, 1-(LM+RM)/FW, BM/FH),
                   right_box=(1-RM/FW, BM/FH, RM/FW, 1-(TM+BM)/FH),
                   left_box=(0, BM/FH, LM/FW, 1-(TM+BM)/FH),
                   )
    if gridspec:
        fig = plt.figure(figsize=(FW, FH), dpi=dpi, **kwargs)
        gs = fig.add_gridspec(RN, CN, wspace=HS/AW, hspace=VS/AH, left=LM/FW, right=1-RM/FW, top=1-TM/FH, bottom=BM/FH)
        return figinfo, fig, gs
    else:
        fig, axes = plt.subplots(RN, CN, figsize=(FW, FH), dpi=dpi, **kwargs)
        fig.subplots_adjust(left=LM/FW, right=1-RM/FW, top=1-TM/FH, bottom=BM/FH, wspace=HS/AW, hspace=VS/AH)
        axes = axes.ravel() if RN * CN > 1 else [axes]
        return figinfo, fig, axes


def plot_resilience(f:callable, extent=(0, 1, 0, 1), gridnum=(1000, 1000), cmap=None, norm=None, ax=None, reset_xylim=True, lw=1):
    """
    绘制二维函数的韧性
    - f: 二维函数，如 lambda x, y: y - 3*y**2 - y**3 + x*y**3
    - extent: 函数定义域 (xmin, xmax, ymin, ymax)
    - gridnum: 网格数量 (xnum, ynum)
    ---
    Example:
    >>> f = lambda x, y: y - 3*y**2 - y**3 + x*y**3
    >>> plot_resilience(f, ax=axes[0], extent=(0, 5, 0, 5))
    >>> f = lambda x, y: np.sin(np.sqrt(x**2 + y ** 2))
    >>> plot_resilience(f, ax=axes[1], extent=(-4*np.pi, 4*np.pi, -4*np.pi, 4*np.pi))
    >>> f = lambda x, y: np.sin(np.sqrt(x ** 2 + y ** 2)) * x - y * np.cos(np.sqrt(x ** 2 + y ** 2))
    >>> plot_resilience(f, ax=axes[2], extent=(-4*np.pi, 4*np.pi, -4*np.pi, 4*np.pi))
    """
    if cmap is None: cmap = mcolors.LinearSegmentedColormap.from_list('my_bwr', ['#74b9ff', 'white', '#ff7675'])
    if norm is None: norm = mcolors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=-1, vmax=1)
    if ax is None: ax = plt.gca()

    def plot1(x, y):
        ax.plot(x, y, 'black', linewidth=lw, linestyle='-', solid_capstyle='round')

    def plot2(x, y):
        ax.plot(x, y, 'gray', linewidth=lw, linestyle=(3, (3, 1)), solid_capstyle='round')

    xmin, xmax, ymin, ymax = extent
    xnum, ynum = gridnum
    x = np.linspace(xmin, xmax, xnum)
    y = np.linspace(ymin, ymax, ynum)
    x, y = np.meshgrid(x, y)
    z = f(x, y)

    ax.imshow(z, origin='lower', aspect='auto', cmap=cmap, norm=norm, zorder=0, extent=(xmin, xmax, ymin, ymax))

    contours = ax.contour(x, y, z, levels=[0], linewidths=0)
    for coll in contours.collections: coll.remove()

    dzdy = np.gradient(z)[0]
    for coll in contours.collections:
        for path in coll.get_paths():
            saved = []
            sign = None
            for (x0, y0) in path.vertices:
                saved.append((x0, y0))
                x_idx = int(np.floor((x0-xmin) / (xmax-xmin) * (xnum - 1)))
                y_idx = int(np.floor((y0-ymin) / (ymax-ymin) * (ynum - 1)))
                if sign is None: sign = np.sign(dzdy[y_idx, x_idx])
                if np.sign(dzdy[y_idx, x_idx] * sign) < 0:
                    if sign < 0: plot1(*np.array(saved).T)
                    else: plot2(*np.array(saved).T)
                    sign *= -1
                    saved = []
                else:
                    pass
            if saved:
                if sign < 0: plot1(*np.array(saved).T)
                else: plot2(*np.array(saved).T)

    if reset_xylim:
        scale = lambda min, max, ratio=0.1: (min - (max - min) * ratio / 2, max + (max - min) * ratio / 2)
        ax.set_xlim(*scale(xmin, xmax))
        ax.set_ylim(*scale(ymin, ymax))


class EqualizeNormalize(mcolors.Normalize):
    """ 按分布而非值进行归一化 """
    def __init__(self, samples, clip=False):
        super().__init__(vmin=samples.min(), vmax=samples.max(), clip=clip)
        hist, bin_edges = np.histogram(samples.flatten(), bins=256, range=(self.vmin, self.vmax), density=True)
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        self.bin_edges = bin_edges
        self.cdf = cdf
    def __call__(self, value, clip=False):
        value = np.array(value)
        return np.ma.masked_array(np.interp(value.flatten(), self.bin_edges[:-1], self.cdf).reshape(value.shape))
    def inverse(self, value):
        value = np.array(value)
        return np.ma.masked_array(np.interp(value.flatten(), self.cdf, self.bin_edges[:-1]).reshape(value.shape))


def plotOD(ax, source:List[str], destination:List[str], flow:List[float], location:Dict[str, Tuple[float, float]],
           linetype:Literal['straight', 'parabola', 'rotated_parabola', 'projected_parabola']='straight', N=100, zorder=10,
           **kwargs):
    """ 绘制OD流量 """
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', ['#0308F8', '#FD0B1B', '#ffff00'], gamma=5.0)
    norm = EqualizeNormalize(flow.values)
    t = np.linspace(0, 1, N)
    ignored = set(list(source) + list(destination)) - set(location.keys())
    if ignored: print(f'\033[33mWarning: {ignored} are not in location\033[0m')
    for (s, d, f) in zip(source, destination, flow):
        if s not in location or d not in location: continue
        p1 = np.array(location[s])[:, np.newaxis] # (2, 1)
        p2 = np.array(location[d])[:, np.newaxis] # (2, 1)
        if linetype == 'straight':
            ax.plot(*(p1 * (1 - t) + p2 * t), lw=0.05+0.1*norm(f), alpha=0.4+0.4*norm(f), color=cmap(norm(f)), zorder=zorder+norm(f), rasterized=kwargs.get('rasterized', False))
        elif linetype == 'parabola':
            y_scale = locals().get('y_scale', np.diff(ax.get_ylim())[0])
            xy = p1 * (1-t) + p2 * t + np.array([[0], [1]]) * 4 * t * (1-t) * norm(f) * y_scale * kwargs.get('scale', 0.1)
            ax.plot(*xy, lw=0.05+0.1*norm(f), alpha=0.4+0.4*norm(f), color=cmap(norm(f)), zorder=zorder+norm(f), rasterized=kwargs.get('rasterized', False))
        elif linetype == 'rotated_parabola':
            height = 0.5 * norm(f)
            C, S = (p2 - p1)[:, 0]
            A = np.array([[C, -S], [S, C]]) / 2
            if kwargs.get('adjust_up', None) and (C < 0): A = -A
            if kwargs.get('adjust_down', None) and (C > 0): A = -A
            xy = A @ np.array([2 * t - 1, height * 4 * t * (1 - t)]) + 0.5 * (p1 + p2)
            ax.plot(*xy, lw=0.05+0.1*norm(f), alpha=0.4+0.4*norm(f), color=cmap(norm(f)), zorder=zorder+norm(f), rasterized=kwargs.get('rasterized', False))
        elif linetype == 'projected_parabola':
            p0 = locals().get('p0', np.array(kwargs.get('p0', [np.mean(ax.get_xlim()), np.mean(ax.get_ylim())]))[:, np.newaxis])
            D = kwargs.get('D', 10)
            xy = p0 + D * (p1 * (1-t) + p2 * t - p0) / (D - 4*t*(1 - t)*(norm(f) + 1))
            ax.plot(*xy, lw=0.05+0.1*norm(f), alpha=0.4+0.4*norm(f), color=cmap(norm(f)), zorder=zorder+norm(f), rasterized=kwargs.get('rasterized', False))
        else:
            raise ValueError(f'Invalid linetype: {linetype}')
    return ax


def clear_svg(path, debug=False):
    """
    matplotlib 生成的 svg 中会使用 <text style="font: 9.8px 'Arial'; text-anchor: middle" x="80.307802" y="193.900483">2020-02-02</text> 的语法，而 Powerpoint 无法识别 font: 9.8px 'Arial'; 的简写记法，只能识别 font-family: 'Arial'; font-size: 9.8px; 的记法。因此需要进行转换。考虑的属性包括：
    - font-size
    - font-family
    - font-weight
    - font-style
    """
    raise DeprecationWarning("现在好像不需要 clear 了，直接导入 PPT 即可")
    from lxml import etree
    tree = etree.parse(path)
    root = tree.getroot()

    for text in root.findall('.//{http://www.w3.org/2000/svg}tspan') + root.findall('.//{http://www.w3.org/2000/svg}text'):
        style = text.attrib.pop('style', '')

        font_size = re.search(r'font:[^;]*\s+(\d+\.?\d*px)(?:\s|$)', style)
        font_family = re.search(r'font:[^;]*\s+\'([^\']*)\'(?:\s|$)', style)
        font_style = re.search(r'font:[^;]*\s+(italic|oblique)(?:\s|$)', style)
        font_weight = re.search(r'font:[^;]*\s+(bold|normal|bolder|lighter)(?:\s|$)', style)
        line_height = re.search(r'line-height:\s+([\d\.]+(px|em|%)?)', style)
        font_variant = re.search(r'font-variant:\s+([\w-]+)', style)
        text_anchor = re.search(r'text-anchor:\s+([\w-]+)', style)

        if font_size: text.set('font-size', font_size.group(1))
        if font_family: text.set('font-family', font_family.group(1))
        if font_style: text.set('font-style', font_style.group(1))
        if font_weight: text.set('font-weight', font_weight.group(1))
        if line_height: text.set('line-height', line_height.group(1))
        if font_variant: text.set('font-variant', font_variant.group(1))
        if text_anchor: text.set('text-anchor', text_anchor.group(1))

        if debug: print(f'"{style}" -> "{text.attrib}"')
    tree.write(path)


def load_font():
    """
    - plt.title("示例图表：数字平方", fontproperties=font, size=15)
    - plt.xlabel("数字", fontproperties=font, size=12)
    - plt.ylabel("平方", fontproperties=font, size=12)
    """
    import requests
    from matplotlib.font_manager import FontProperties

    url = "https://ghp.ci/https://github.com/notofonts/noto-cjk/blob/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf"
    font_path = "/tmp/SimHei.otf"
    if not os.path.exists(font_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(font_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception("Failed to download the font.")
    font = FontProperties(fname=font_path)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_path]  # 这里指定.otf文件路径
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
    return font


# 计算两个 box 的重叠面积
def _overlapping_area(bbox1, bbox2):
    x0 = max(bbox1.x0, bbox2.x0)
    x1 = min(bbox1.x1, bbox2.x1)
    y0 = max(bbox1.y0, bbox2.y0)
    y1 = min(bbox1.y1, bbox2.y1)
    return max(0, x1 - x0) * max(0, y1 - y0)


# 计算 point 到 box 的最近距离
def _distance_to_box(point, box):
    x = max(box.x0, min(point[0], box.x1))
    y = max(box.y0, min(point[1], box.y1))
    return np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)


# 调整文本位置的函数
def adjust_text(texts, ax, step=0.01, max_iterations=100, mode="xy"):
    raw_pos = np.array([text.get_position() for text in texts], dtype=float)
    cur_pos = raw_pos.copy()

    overlap = np.zeros((len(texts), len(texts)))

    def update(i, utri_only=False):
        bbox1 = texts[i].get_window_extent(renderer=ax.figure.canvas.get_renderer())
        for j, text2 in enumerate(texts):
            if j == i:
                continue
            if utri_only and j <= i:
                continue
            bbox2 = text2.get_window_extent(renderer=ax.figure.canvas.get_renderer())
            overlap[i, j] = overlap[j, i] = _overlapping_area(bbox1, bbox2)

    def normalize(v):
        return v / np.linalg.norm(v).clip(1e-6)

    # 初始化
    for i in range(len(texts)):
        update(i, utri_only=True)

    for iteration in range(max_iterations):
        i, j = random.choice(list(np.stack(np.nonzero(overlap), axis=-1)))

        # i 对 j 的排斥
        F1 = cur_pos[j] - cur_pos[i]

        # raw_pos[j] 对 j 的吸引
        bbox = texts[j].get_window_extent(renderer=ax.figure.canvas.get_renderer())
        F2 = (
            normalize(raw_pos[j] - cur_pos[j])
            * _distance_to_box(raw_pos[j], bbox)
            * 0.001
        )
        # F2 = 0.0

        # 随机扰动
        F3 = np.random.randn(2)

        # 合力
        F = F1 + F2 + F3
        if mode == "x":
            F[1] = 0
        if mode == "y":
            F[0] = 0

        # 更新位置
        cur_pos[j] += F * step
        texts[j].set_position(cur_pos[j])
        update(j)
        if (overlap == 0).all():
            break


def use_chinese_font(fontpath="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    font_prop = FontProperties(fname=fontpath)
    rcParams["font.family"] = font_prop.get_name()
    rcParams["axes.unicode_minus"] = False


def merge_axes(axes, fig=None):
    if fig is None: fig = axes[0].figure
    xmin = min(ax.get_position().bounds[0] for ax in axes)  # axes[0, 2] 的左上角坐标
    xmax = max(ax.get_position().bounds[0] + ax.get_position().bounds[2] for ax in axes)  # axes[1, 3] 的右下角坐标
    ymin = min(ax.get_position().bounds[1] for ax in axes)  # axes[0, 2] 的左上角坐标
    ymax = max(ax.get_position().bounds[1] + ax.get_position().bounds[3] for ax in axes)  # axes[1, 3] 的右下角坐标
    merged_ax = fig.figure.add_axes([xmin, ymin, xmax - xmin, ymax - ymin])
    
    # 如果 axes[0] 被设置与其他的 ax 共享坐标轴，则需要设置 merged_ax 也共享坐标轴
    # if axes[0].get_shared_x_axes() is not None:
    #     merged_ax.get_shared_x_axes().join(merged_ax, *axes)
    # if axes[0].get_shared_y_axes() is not None:
    #     merged_ax.get_shared_y_axes().join(merged_ax, *axes)

    for ax in axes:
        fig.delaxes(ax)
    return merged_ax
