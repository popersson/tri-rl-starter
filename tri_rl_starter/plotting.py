"""Matplotlib plotting helpers for triangular meshes."""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.tri import Triangulation
except ImportError:
    plt = None
    LineCollection = None
    BoundaryNorm = None
    ListedColormap = None
    Triangulation = None


DEGREE_COLORS = (
    "#332288",  # -4 and below
    "#0072B2",
    "#009E73",
    "#56B4E9",
    "#111111",  # 0, mostly hidden by neutral dots
    "#F0E442",
    "#E69F00",
    "#D55E00",
    "#CC79A7",  # +4 and above
)


def _enable_ipython_gui():
    try:
        get_ipython
    except NameError:
        return False

    ip = get_ipython()
    if ip is None or not hasattr(ip, "enable_gui"):
        return False

    backend = plt.get_backend().lower()
    if "qt" in backend:
        gui = "qt"
    elif "tk" in backend:
        gui = "tk"
    elif "wx" in backend:
        gui = "wx"
    elif "gtk4" in backend:
        gui = "gtk4"
    elif "gtk3" in backend or "gtk" in backend:
        gui = "gtk3"
    elif "macosx" in backend:
        gui = "osx"
    else:
        return False

    try:
        ip.enable_gui(gui)
    except Exception:
        return False
    return True


def _degree_mode(degree_coloring):
    if isinstance(degree_coloring, str):
        return degree_coloring
    return "discrete" if degree_coloring else "none"


def _plot_discrete_degree_points(ax, points, diff, neutral,
                                 degree_range=(-3, 5)):
    if ListedColormap is None or BoundaryNorm is None:
        raise ImportError("discrete degree coloring requires matplotlib colors")

    values = np.rint(diff).astype(int)
    nonneutral = ~neutral
    if not np.any(nonneutral):
        return

    lo = min(int(degree_range[0]), int(values[nonneutral].min()))
    hi = max(int(degree_range[1]), int(values[nonneutral].max()))
    levels = np.arange(lo, hi + 1, dtype=int)
    base_levels = np.arange(-4, 5, dtype=int)
    color_lookup = {
        level: DEGREE_COLORS[
            int(np.clip(level, base_levels[0], base_levels[-1])
                - base_levels[0])
        ]
        for level in levels
    }
    colors = [color_lookup[level] for level in levels]
    cmap = ListedColormap(colors)
    bounds = np.arange(lo - 0.5, hi + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    clipped = np.clip(values[nonneutral], lo, hi)
    sc = ax.scatter(
        points[nonneutral, 0], points[nonneutral, 1],
        s=34, c=clipped, cmap=cmap, norm=norm,
        edgecolors="0.10", linewidths=0.25,
        zorder=5)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.78, ticks=levels)
    cbar.set_label("degree - ideal")


def plot_mesh(mesh, ax=None, degree_coloring=True, show=True,
              annotate_nodes=False, degree_range=(-3, 5), block=False):
    if plt is None or LineCollection is None or Triangulation is None:
        raise ImportError("tri_rl_starter plotting requires matplotlib")

    if ax is None:
        fig = plt.gcf()
        fig.clf()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
        for other in list(fig.axes):
            if other is not ax:
                fig.delaxes(other)
        ax.clear()

    points = mesh.points
    tri = Triangulation(points[:, 0], points[:, 1], mesh.triangles)
    ax.set_aspect("equal", adjustable="box")
    ax.triplot(tri, color="0.72", linewidth=0.65)

    if len(mesh.boundary_edges):
        ax.add_collection(LineCollection(
            points[mesh.boundary_edges],
            colors="0.05",
            linewidths=1.8,
            zorder=3))

    mode = _degree_mode(degree_coloring)
    if mode != "none":
        diff = mesh.degree_deviation()["signed"]
        neutral = np.abs(diff) <= 1.0e-12
        if np.any(neutral):
            ax.scatter(
                points[neutral, 0], points[neutral, 1],
                s=8, c="0.05", linewidths=0, zorder=4)
        if np.any(~neutral):
            if mode == "discrete":
                _plot_discrete_degree_points(
                    ax, points, diff, neutral, degree_range=degree_range)
            elif mode == "continuous":
                vmax = max(float(np.abs(diff[~neutral]).max()), 1.0)
                sc = ax.scatter(
                    points[~neutral, 0], points[~neutral, 1],
                    s=34, c=diff[~neutral], cmap="coolwarm",
                    vmin=-vmax, vmax=vmax,
                    edgecolors="0.10", linewidths=0.25,
                    zorder=5)
                plt.colorbar(sc, ax=ax, shrink=0.78, label="degree - ideal")
            else:
                raise ValueError(
                    "degree_coloring must be 'discrete', 'continuous', "
                    "'none', True, or False")
    else:
        ax.scatter(points[:, 0], points[:, 1], s=8, c="0.05", zorder=4)

    if len(mesh.corner_nodes):
        corners = mesh.corner_nodes
        ax.scatter(
            points[corners, 0], points[corners, 1],
            s=55, facecolors="none", edgecolors="0.0",
            linewidths=1.0, zorder=6)

    if annotate_nodes:
        span = points.max(axis=0) - points.min(axis=0)
        offset = 0.012 * max(float(span.max()), 1.0)
        for i, point in enumerate(points):
            ax.annotate(str(i), point + offset, fontsize=7)

    ax.autoscale_view()
    if show:
        if not block:
            _enable_ipython_gui()
        plt.show(block=block)
        if not block:
            plt.pause(0.001)
    return ax
