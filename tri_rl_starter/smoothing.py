"""Simple sample smoothers for triangular meshes."""

from collections import defaultdict

import numpy as np

from .metrics import triangle_areas


AREA_TOL = 1.0e-14


def _trace_boundary_loops(boundary_edges):
    graph = defaultdict(set)
    unused = set()
    for a, b in np.asarray(boundary_edges, dtype=int).reshape(-1, 2):
        a = int(a)
        b = int(b)
        if a == b:
            continue
        graph[a].add(b)
        graph[b].add(a)
        unused.add(frozenset((a, b)))
    if any(len(neighbors) != 2 for neighbors in graph.values()):
        return []

    loops = []
    while unused:
        edge = next(iter(unused))
        start, current = tuple(edge)
        previous = start
        loop = [start, current]
        unused.remove(edge)
        while current != start:
            choices = [node for node in graph[current] if node != previous]
            if not choices:
                return []
            nxt = choices[0]
            edge = frozenset((current, nxt))
            if edge not in unused:
                if nxt == start:
                    break
                return []
            unused.remove(edge)
            previous, current = current, nxt
            if current != start:
                if current in loop:
                    return []
                loop.append(current)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _cyclic_chain(loop, start_index, stop_index):
    n = len(loop)
    chain = [loop[start_index]]
    i = start_index
    while i != stop_index:
        i = (i + 1) % n
        chain.append(loop[i])
    return chain


def _boundary_chains(boundary_edges, corner_nodes):
    corners = set(int(node) for node in np.asarray(corner_nodes, dtype=int))
    chains = []
    for loop in _trace_boundary_loops(boundary_edges):
        corner_indices = [
            i for i, node in enumerate(loop)
            if int(node) in corners
        ]
        if len(corner_indices) < 2:
            continue
        for k, start_index in enumerate(corner_indices):
            stop_index = corner_indices[(k + 1) % len(corner_indices)]
            chain = _cyclic_chain(loop, start_index, stop_index)
            interior = tuple(
                int(node) for node in chain[1:-1]
                if int(node) not in corners
            )
            if interior:
                chains.append((int(chain[0]), int(chain[-1]), interior))
    return chains


def equispace_boundary_points(mesh, points=None):
    """Equispace non-corner boundary nodes along each polygon side."""
    out = (
        np.asarray(mesh.points, dtype=float).copy()
        if points is None else np.asarray(points, dtype=float).copy()
    )
    for start, stop, nodes in _boundary_chains(
        mesh.boundary_edges, mesh.corner_nodes
    ):
        a = out[start]
        b = out[stop]
        count = len(nodes)
        for i, node in enumerate(nodes, start=1):
            t = i / float(count + 1)
            out[node] = (1.0 - t) * a + t * b
    return out


def _geometry_ok(points, triangles):
    return bool(np.all(triangle_areas(points, triangles, signed=True) > AREA_TOL))


def smooth_points(mesh, iterations=10, omega=0.5, check_inversion=True):
    """Return smoothed points for ``mesh`` without modifying it.

    This intentionally modest sample smoother first redistributes non-corner
    boundary nodes uniformly along each corner-to-corner polygon side. It then
    applies damped graph-Laplacian smoothing to interior nodes only.
    """
    points0 = np.asarray(mesh.points, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)
    points = equispace_boundary_points(mesh, points0)
    if check_inversion and not _geometry_ok(points, triangles):
        points = points0.copy()

    boundary = set(int(node) for node in mesh.boundary_nodes)
    movable = [
        node for node in range(len(points))
        if node not in boundary
    ]

    iterations = max(int(iterations), 0)
    omega = float(omega)
    for _ in range(iterations):
        trial = points.copy()
        for node in movable:
            neighbors = mesh.vertex_neighbors[node]
            if not neighbors:
                continue
            target = points[np.asarray(neighbors, dtype=int)].mean(axis=0)
            trial[node] = (1.0 - omega) * points[node] + omega * target

        if not check_inversion or _geometry_ok(trial, triangles):
            points = trial
            continue

        accepted = False
        step = omega
        for _attempt in range(8):
            step *= 0.5
            trial = points.copy()
            for node in movable:
                neighbors = mesh.vertex_neighbors[node]
                if not neighbors:
                    continue
                target = points[np.asarray(neighbors, dtype=int)].mean(axis=0)
                trial[node] = (1.0 - step) * points[node] + step * target
            if _geometry_ok(trial, triangles):
                points = trial
                accepted = True
                break
        if not accepted:
            break
    return points


def smooth_mesh(mesh, iterations=10, omega=0.5, check_inversion=True):
    """Return a new ``TriMesh`` with smoothed point coordinates."""
    return mesh.with_points(
        smooth_points(
            mesh,
            iterations=iterations,
            omega=omega,
            check_inversion=check_inversion))
