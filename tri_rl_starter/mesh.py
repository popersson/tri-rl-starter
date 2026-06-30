"""Pure triangular mesh data structure for RL experiments."""

from dataclasses import dataclass

import numpy as np

from .metrics import degree_deviation, orient_triangles_ccw, quality_summary


def _unique_edges(triangles):
    triangles = np.asarray(triangles, dtype=int)
    edges = np.vstack((
        triangles[:, [1, 2]],
        triangles[:, [2, 0]],
        triangles[:, [0, 1]],
    ))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _edge_counts(triangles):
    counts = {}
    for it, tri in enumerate(np.asarray(triangles, dtype=int)):
        for local_edge, (i, j) in enumerate(((1, 2), (2, 0), (0, 1))):
            edge = tuple(sorted((int(tri[i]), int(tri[j]))))
            counts.setdefault(edge, []).append((it, local_edge))
    return counts


def _triangle_adjacency(triangles):
    triangles = np.asarray(triangles, dtype=int)
    t2t = np.full((len(triangles), 3), -1, dtype=int)
    t2n = np.full((len(triangles), 3), -1, dtype=int)
    for adjacent in _edge_counts(triangles).values():
        if len(adjacent) != 2:
            continue
        (t0, e0), (t1, e1) = adjacent
        t2t[t0, e0] = t1
        t2n[t0, e0] = e1
        t2t[t1, e1] = t0
        t2n[t1, e1] = e0
    return t2t, t2n


def _vertex_neighbors(triangles, npoints):
    graph = [set() for _ in range(npoints)]
    for a, b in _unique_edges(triangles):
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    return tuple(tuple(sorted(neighbors)) for neighbors in graph)


def _degrees(triangles, npoints):
    edges = _unique_edges(triangles)
    degrees = np.zeros(npoints, dtype=int)
    if len(edges):
        np.add.at(degrees, edges[:, 0], 1)
        np.add.at(degrees, edges[:, 1], 1)
    return degrees


def _angle_sums(points, triangles):
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    out = np.zeros(len(points), dtype=float)
    xy = points[triangles]
    for local in range(3):
        a = xy[:, local]
        b = xy[:, (local + 1) % 3]
        c = xy[:, (local + 2) % 3]
        u = b - a
        v = c - a
        dot = np.einsum("ij,ij->i", u, v)
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        denom = np.maximum(nu * nv, 1.0e-300)
        angle = np.arccos(np.clip(dot / denom, -1.0, 1.0))
        np.add.at(out, triangles[:, local], angle)
    return out


def _ideal_degrees(points, triangles, boundary_nodes,
                   round_boundary=True, alpha_degrees=60.0):
    angles = _angle_sums(points, triangles) * 180.0 / np.pi
    ideal = np.full(len(points), 360.0 / float(alpha_degrees), dtype=float)
    boundary_nodes = np.asarray(boundary_nodes, dtype=int)
    if len(boundary_nodes):
        raw = angles[boundary_nodes] / float(alpha_degrees) + 1.0
        if round_boundary:
            raw = np.floor(raw + 0.5)
        ideal[boundary_nodes] = np.maximum(raw, 2.0)
    return ideal


def _minimum_degrees(points, triangles, boundary_nodes, angle_tol=5.0):
    angles = _angle_sums(points, triangles) * 180.0 / np.pi
    minimum = np.full(len(points), 3, dtype=int)
    max_angle = 180.0 - float(angle_tol)
    for node in np.asarray(boundary_nodes, dtype=int):
        theta = angles[node]
        if theta < max_angle:
            minimum[node] = 2
        elif theta < 2.0 * max_angle:
            minimum[node] = 3
        else:
            minimum[node] = 4
    return minimum


@dataclass
class TriMesh:
    points: np.ndarray
    triangles: np.ndarray
    boundary_edges: np.ndarray
    corner_nodes: np.ndarray
    t2t: np.ndarray
    t2n: np.ndarray
    vertex_neighbors: tuple
    boundary_nodes: np.ndarray
    ideal_degrees: np.ndarray
    minimum_degrees: np.ndarray

    def __init__(self, points, triangles, boundary_edges,
                 corner_nodes=None, round_boundary=True, angle_tol=5.0,
                 ideal_degrees=None, minimum_degrees=None):
        self.points = np.asarray(points, dtype=float).copy()
        self.triangles = orient_triangles_ccw(
            self.points, np.asarray(triangles, dtype=int))
        self.boundary_edges = np.asarray(boundary_edges, dtype=int).reshape(-1, 2)
        if corner_nodes is None:
            corner_nodes = np.unique(self.boundary_edges.ravel())
        self.corner_nodes = np.unique(np.asarray(corner_nodes, dtype=int))

        self.t2t, self.t2n = _triangle_adjacency(self.triangles)
        self.vertex_neighbors = _vertex_neighbors(self.triangles, len(self.points))
        if len(self.boundary_edges):
            self.boundary_nodes = np.unique(self.boundary_edges.ravel())
        else:
            counts = _edge_counts(self.triangles)
            self.boundary_nodes = np.unique([
                node for edge, adj in counts.items() if len(adj) == 1
                for node in edge
            ]).astype(int)

        if ideal_degrees is None:
            self.ideal_degrees = _ideal_degrees(
                self.points, self.triangles, self.boundary_nodes,
                round_boundary=round_boundary)
        else:
            self.ideal_degrees = np.asarray(ideal_degrees, dtype=float).copy()
            if len(self.ideal_degrees) != len(self.points):
                raise ValueError("ideal_degrees must have one value per point")

        if minimum_degrees is None:
            self.minimum_degrees = _minimum_degrees(
                self.points, self.triangles, self.boundary_nodes,
                angle_tol=angle_tol)
        else:
            self.minimum_degrees = np.asarray(minimum_degrees, dtype=int).copy()
            if len(self.minimum_degrees) != len(self.points):
                raise ValueError("minimum_degrees must have one value per point")

    @property
    def npoints(self):
        return int(len(self.points))

    @property
    def nelements(self):
        return int(len(self.triangles))

    def degrees(self):
        return _degrees(self.triangles, len(self.points))

    def degree_deviation(self):
        return degree_deviation(self.degrees(), self.ideal_degrees)

    def minimum_degree_deficit(self):
        deficit = np.maximum(self.minimum_degrees - self.degrees(), 0)
        return {
            "deficit": deficit,
            "count": int(np.count_nonzero(deficit)),
            "total": int(deficit.sum()),
            "max": int(deficit.max()) if len(deficit) else 0,
        }

    def quality_summary(self):
        return quality_summary(self.points, self.triangles)

    def with_points(self, points):
        """Return a mesh with identical topology and new point coordinates."""
        return type(self)(
            points,
            self.triangles,
            self.boundary_edges,
            corner_nodes=self.corner_nodes)

    def with_triangles(self, triangles, preserve_degree_targets=True):
        """Return a mesh with identical vertices and new triangle connectivity."""
        kwargs = {}
        if preserve_degree_targets:
            kwargs["ideal_degrees"] = self.ideal_degrees
            kwargs["minimum_degrees"] = self.minimum_degrees
        return type(self)(
            self.points,
            triangles,
            self.boundary_edges,
            corner_nodes=self.corner_nodes,
            **kwargs)

    def smoothed(self, iterations=10, omega=0.5, check_inversion=True):
        """Return a simply smoothed copy of this mesh."""
        from .smoothing import smooth_mesh

        return smooth_mesh(
            self,
            iterations=iterations,
            omega=omega,
            check_inversion=check_inversion)

    def summary(self):
        dev = self.degree_deviation()
        qual = self.quality_summary()
        min_def = self.minimum_degree_deficit()
        return {
            "points": self.npoints,
            "triangles": self.nelements,
            "boundary_nodes": int(len(self.boundary_nodes)),
            "corner_nodes": int(len(self.corner_nodes)),
            "degree_l1": dev["l1"],
            "degree_l2": dev["l2"],
            "degree_linf": dev["linf"],
            "quality_min": qual["min"],
            "quality_mean": qual["mean"],
            "quality_p10": qual["p10"],
            "min_deficit_count": min_def["count"],
            "min_deficit_total": min_def["total"],
        }
