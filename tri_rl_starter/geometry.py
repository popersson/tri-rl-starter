"""Random test geometries and Triangle-based mesh generation."""

import math

import numpy as np


GEOM_TOL = 1.0e-12
CASE_NAMES = (
    "regular_polygon",
    "random_polygon",
    "polygon_hole",
    "grid_orth",
    "grid_oct",
)


def _triangle_module():
    try:
        import triangle
    except ImportError as exc:
        raise ImportError(
            "tri_rl_starter needs the optional 'triangle' package. "
            "Install it with: python -m pip install triangle"
        ) from exc
    return triangle


def _polygon_points(polygon):
    p = np.asarray(polygon, dtype=float)
    if p.ndim != 2 or 2 not in p.shape:
        raise ValueError("polygon points must have shape (n, 2) or (2, n)")
    p = p.copy() if p.shape[1] == 2 else p.T.copy()
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3:
        raise ValueError("a polygon needs at least 3 distinct vertices")
    return p


def _hole_points(holes):
    if holes is None:
        return np.empty((0, 2), dtype=float)
    h = np.asarray(holes, dtype=float)
    if h.size == 0:
        return np.empty((0, 2), dtype=float)
    if h.ndim == 1:
        if h.size % 2:
            raise ValueError("holes must contain x,y coordinate pairs")
        return h.reshape(-1, 2)
    if h.ndim != 2 or 2 not in h.shape:
        raise ValueError("holes must have shape (m, 2), (2, m), or (2,)")
    return h.copy() if h.shape[1] == 2 else h.T.copy()


def polygon_segments(nvertices, offset=0):
    nodes = np.arange(offset, offset + nvertices, dtype=int)
    return np.column_stack([nodes, np.roll(nodes, -1)])


def _signed_area(polygon):
    p = _polygon_points(polygon)
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orient_polygon(polygon, ccw=True):
    p = _polygon_points(polygon)
    if (_signed_area(p) > 0.0) != bool(ccw):
        p = p[::-1].copy()
    return p


def _simplify_polygon(polygon, tol=GEOM_TOL):
    p = _polygon_points(polygon)
    changed = True
    while changed and len(p) > 3:
        changed = False
        keep = []
        n = len(p)
        for i in range(n):
            a = p[(i - 1) % n]
            b = p[i]
            c = p[(i + 1) % n]
            if np.linalg.norm(b - a) <= tol:
                changed = True
                continue
            u = b - a
            v = c - b
            cross = u[0] * v[1] - u[1] * v[0]
            if abs(cross) <= tol:
                changed = True
                continue
            keep.append(b)
        if len(keep) < 3:
            break
        p = np.asarray(keep, dtype=float)
    return p


def triangle_command(hmax=None, min_angle=28.6, quality=True, extra=""):
    """Build a Triangle command string.

    Python's Triangle wrapper does not expose Julia Triangulate's
    ``triunsuitable!`` callback, so ``hmax`` is approximated through Triangle's
    global maximum-area refinement option.
    """
    cmd = "pQ"
    if quality:
        cmd += f"q{float(min_angle):g}"
    if hmax is not None and math.isfinite(float(hmax)):
        max_area = math.sqrt(3.0) * float(hmax) * float(hmax) / 4.0
        cmd += f"a{max_area:.17g}"
    return cmd + str(extra)


def polytrimesh(polygons, holes=None, hmax=None, min_angle=28.6,
                quality=True, cmd=None):
    """Triangulate polygonal regions with optional uniform refinement.

    Args:
        polygons: exterior polygon followed by optional hole polygons.
        holes: one point inside each hole.
        hmax: approximate target maximum edge length via a max-area constraint.
        min_angle: Triangle quality-refinement angle when ``quality`` is true.
        quality: add Triangle's quality-refinement switch.
        cmd: explicit Triangle command string. If provided, it overrides
            ``hmax``, ``min_angle``, and ``quality``.

    Returns:
        ``p, t, e0, corner_nodes``.
    """
    if isinstance(polygons, np.ndarray):
        polygons = [polygons]

    points = []
    segments = []
    offset = 0
    for polygon in polygons:
        p = _polygon_points(polygon)
        points.append(p)
        segments.append(polygon_segments(len(p), offset))
        offset += len(p)

    if not points:
        raise ValueError("polytrimesh requires at least one polygon")

    pointlist = np.vstack(points)
    segmentlist = np.vstack(segments)
    triin = {
        "vertices": pointlist,
        "segments": segmentlist,
    }

    holelist = _hole_points(holes)
    if len(holelist):
        triin["holes"] = holelist

    triangle = _triangle_module()
    command = triangle_command(
        hmax=hmax, min_angle=min_angle, quality=quality) if cmd is None else cmd
    triout = triangle.triangulate(triin, command)
    if "triangles" not in triout:
        raise RuntimeError("Triangle did not return any triangles")

    p = np.asarray(triout["vertices"], dtype=float)
    t = np.asarray(triout["triangles"], dtype=int)
    e0 = np.asarray(triout.get("segments", segmentlist), dtype=int)
    corner_nodes = np.arange(len(pointlist), dtype=int)
    return p, t, e0, corner_nodes


def random_polygon(nvertices=20, seed=None):
    if nvertices < 3:
        raise ValueError("random_polygon requires at least 3 vertices")
    phi = 2.0 * np.pi * np.arange(1, nvertices + 1) / nvertices
    rng = np.random.default_rng(seed)
    r = 0.5 + rng.random(nvertices)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


def regular_polygon(nvertices=20, radius=1.0):
    if nvertices < 3:
        raise ValueError("regular_polygon requires at least 3 vertices")
    phi = 2.0 * np.pi * np.arange(nvertices) / nvertices
    return float(radius) * np.column_stack([np.cos(phi), np.sin(phi)])


def _trace_grid_boundary(edges):
    outgoing = {}
    for a, b in edges:
        outgoing.setdefault(a, []).append(b)
    if any(len(targets) != 1 for targets in outgoing.values()):
        raise ValueError("grid-cell union has a pinched boundary")

    unused = set(edges)
    loops = []
    while unused:
        start = min(unused)[0]
        loop = [start]
        current = start
        while True:
            targets = outgoing.get(current)
            if targets is None or len(targets) != 1:
                raise ValueError("grid-cell boundary is not traceable")
            nxt = targets[0]
            edge = (current, nxt)
            if edge not in unused:
                raise ValueError("grid-cell boundary repeats an edge")
            unused.remove(edge)
            if nxt == start:
                break
            if nxt in loop:
                raise ValueError("grid-cell boundary self-touches")
            loop.append(nxt)
            current = nxt
        loops.append(np.asarray(loop, dtype=float))
    return loops


def _grid_boundary_loops(cells, simplify=True):
    cells = {tuple(map(int, cell)) for cell in cells}
    if not cells:
        raise ValueError("at least one grid cell is required")

    edges = set()
    for i, j in cells:
        corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
        for a, b in zip(corners, corners[1:] + corners[:1]):
            reverse = (b, a)
            if reverse in edges:
                edges.remove(reverse)
            else:
                edges.add((a, b))

    loops = _trace_grid_boundary(edges)
    if simplify:
        loops = [_simplify_polygon(loop) for loop in loops]
    return loops


def _cells_to_polygon(cells):
    loops = _grid_boundary_loops(cells, simplify=True)
    loops = sorted(loops, key=lambda loop: abs(_signed_area(loop)), reverse=True)
    outer = [_orient_polygon(loop, ccw=True)
             for loop in loops if _signed_area(loop) > 0.0]
    if len(outer) != 1:
        raise ValueError("grid-cell union must have one exterior boundary")
    return outer[0]


def _random_connected_cells(nblocks, rng):
    cells = {(0, 0)}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    shapes = ((1, 1), (2, 1), (1, 2))
    target = max(int(nblocks), 1)
    for _ in range(target - 1):
        for _attempt in range(200):
            w, h = shapes[int(rng.integers(len(shapes)))]
            base = tuple(tuple(cell) for cell in cells)[int(rng.integers(len(cells)))]
            direction = directions[int(rng.integers(len(directions)))]
            i, j = base
            if direction == (1, 0):
                x = i + 1
                y = j - int(rng.integers(h))
            elif direction == (-1, 0):
                x = i - w
                y = j - int(rng.integers(h))
            elif direction == (0, 1):
                x = i - int(rng.integers(w))
                y = j + 1
            else:
                x = i - int(rng.integers(w))
                y = j - h
            block = {(x + dx, y + dy) for dx in range(w) for dy in range(h)}
            if block - cells:
                cells.update(block)
                break
        else:
            raise RuntimeError("could not grow connected grid-cell union")
    return cells


def _chamfer_convex_corners(polygon, rng, probability=0.4):
    p = _orient_polygon(_simplify_polygon(polygon), ccw=True)
    if probability <= 0.0:
        return p

    q = 2.0 * p
    n = len(q)
    mask = rng.random(n) < probability
    for i in range(n):
        edge_length = np.linalg.norm(q[(i + 1) % n] - q[i], ord=1)
        if mask[i] and mask[(i + 1) % n] and edge_length <= 2.0 + GEOM_TOL:
            mask[(i + 1) % n] = False

    out = []
    for i in range(n):
        a = q[(i - 1) % n]
        b = q[i]
        c = q[(i + 1) % n]
        u = b - a
        v = c - b
        cross = u[0] * v[1] - u[1] * v[0]
        if mask[i] and cross > GEOM_TOL:
            out.append(b - np.sign(u))
            out.append(b + np.sign(v))
        else:
            out.append(b)
    return _simplify_polygon(np.asarray(out, dtype=float))


def random_grid_polygon(nblocks=25, seed=None, chamfer_probability=0.4):
    rng = np.random.default_rng(seed)
    cells = _random_connected_cells(nblocks, rng)
    polygon = _cells_to_polygon(cells)
    return _chamfer_convex_corners(
        polygon, rng, probability=chamfer_probability)


def case_polygons(case="random_polygon", seed=None, nvertices=20,
                  nblocks=25, np_out=20, np_in=8,
                  chamfer_probability=0.4, inner_scale=1.0 / 3.0):
    """Return ``polygons, holes`` for a named random geometry."""
    if case in ("random", "poly"):
        case = "random_polygon"
    elif case in ("regular", "circle"):
        case = "regular_polygon"
    elif case in ("hole", "poly_hole"):
        case = "polygon_hole"
    elif case in ("grid", "orthogonal"):
        case = "grid_orth"
    elif case in ("oct", "octilinear"):
        case = "grid_oct"

    rng = np.random.default_rng(seed)
    if case == "regular_polygon":
        return [regular_polygon(nvertices)], None
    if case == "random_polygon":
        return [random_polygon(nvertices, seed=seed)], None
    if case == "polygon_hole":
        outer = random_polygon(
            np_out, seed=int(rng.integers(2**31)))
        inner = float(inner_scale) * random_polygon(
            np_in, seed=int(rng.integers(2**31)))
        return [outer, inner], np.array([[0.0, 0.0]])
    if case == "grid_orth":
        return [
            random_grid_polygon(
                nblocks=nblocks, seed=seed, chamfer_probability=0.0)
        ], None
    if case == "grid_oct":
        return [
            random_grid_polygon(
                nblocks=nblocks, seed=seed,
                chamfer_probability=chamfer_probability)
        ], None
    known = ", ".join(CASE_NAMES)
    raise ValueError(f"unknown case {case!r}; known cases: {known}")


def generate_case_arrays(case="random_polygon", seed=None, nvertices=20,
                         nblocks=25, np_out=20, np_in=8,
                         chamfer_probability=0.4,
                         inner_scale=1.0 / 3.0,
                         hmax=None, min_angle=28.6,
                         quality=True, cmd=None):
    polygons, holes = case_polygons(
        case=case,
        seed=seed,
        nvertices=nvertices,
        nblocks=nblocks,
        np_out=np_out,
        np_in=np_in,
        chamfer_probability=chamfer_probability,
        inner_scale=inner_scale,
    )
    return polytrimesh(
        polygons,
        holes=holes,
        hmax=hmax,
        min_angle=min_angle,
        quality=quality,
        cmd=cmd,
    )


def generate_case_mesh(*args, **kwargs):
    from .mesh import TriMesh

    p, t, e0, corner_nodes = generate_case_arrays(*args, **kwargs)
    return TriMesh(p, t, e0, corner_nodes=corner_nodes)
