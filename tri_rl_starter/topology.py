"""Topology-only triangular mesh actions for RL environments."""

from dataclasses import dataclass

import numpy as np

from .mesh import _edge_counts, _unique_edges


@dataclass(frozen=True)
class FlipCandidate:
    """A valid 2-to-2 triangular edge flip."""

    edge: tuple[int, int]
    triangles: tuple[int, int]
    opposite_nodes: tuple[int, int]

    @property
    def new_edge(self):
        return tuple(sorted(self.opposite_nodes))


def canonical_edge(edge):
    """Return an undirected edge as a sorted pair of Python ints."""
    edge = tuple(int(node) for node in edge)
    if len(edge) != 2:
        raise ValueError("edge must contain exactly two node indices")
    if edge[0] == edge[1]:
        raise ValueError("edge endpoints must be distinct")
    return tuple(sorted(edge))


def mesh_edges(mesh, directed=False):
    """Return unique mesh edges, optionally in both directed orientations."""
    edges = _unique_edges(mesh.triangles)
    if not directed:
        return edges
    if len(edges) == 0:
        return np.empty((0, 2), dtype=int)
    return np.vstack((edges, edges[:, ::-1]))


def _candidate_from_adjacency(mesh, edge, adjacent, edge_set):
    if len(adjacent) != 2:
        return None

    (t0, local0), (t1, local1) = adjacent
    c = int(mesh.triangles[t0, local0])
    d = int(mesh.triangles[t1, local1])
    if c == d:
        return None

    new_edge = tuple(sorted((c, d)))
    if new_edge in edge_set:
        return None

    return FlipCandidate(
        edge=tuple(edge),
        triangles=(int(t0), int(t1)),
        opposite_nodes=(c, d))


def flip_candidates(mesh):
    """Return all topologically valid interior edge flips.

    This is deliberately geometry-free: it does not check triangle shape or
    inversion. It only preserves a valid simplicial connectivity by requiring
    an interior edge and disallowing duplicate new diagonals.
    """
    counts = _edge_counts(mesh.triangles)
    edge_set = set(counts)
    candidates = []
    for edge, adjacent in counts.items():
        candidate = _candidate_from_adjacency(
            mesh, edge=edge, adjacent=adjacent, edge_set=edge_set)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda candidate: candidate.edge))


def flippable_edges(mesh):
    """Return valid flip edges as an ``(n, 2)`` integer array."""
    candidates = flip_candidates(mesh)
    if not candidates:
        return np.empty((0, 2), dtype=int)
    return np.asarray([candidate.edge for candidate in candidates], dtype=int)


def find_flip_candidate(mesh, edge):
    """Return the flip candidate for ``edge`` or raise ``ValueError``."""
    edge = canonical_edge(edge)
    counts = _edge_counts(mesh.triangles)
    adjacent = counts.get(edge)
    if adjacent is None:
        raise ValueError(f"edge {edge} is not present in the mesh")
    candidate = _candidate_from_adjacency(
        mesh, edge=edge, adjacent=adjacent, edge_set=set(counts))
    if candidate is None:
        raise ValueError(f"edge {edge} is not a valid topological flip")
    return candidate


def degree_after_flip(mesh, edge):
    """Return vertex degrees after flipping ``edge`` without changing the mesh."""
    candidate = find_flip_candidate(mesh, edge)
    degrees = mesh.degrees()
    a, b = candidate.edge
    c, d = candidate.opposite_nodes
    out = degrees.copy()
    out[[a, b]] -= 1
    out[[c, d]] += 1
    return out


def degree_l1_after_flip(mesh, edge):
    """Return degree L1 score after a candidate flip using local degree updates."""
    degrees = degree_after_flip(mesh, edge)
    return float(np.abs(degrees - mesh.ideal_degrees).sum())


def flip_candidate(mesh, candidate, preserve_degree_targets=True):
    """Return a new mesh obtained by applying a precomputed flip candidate."""
    a, b = candidate.edge
    c, d = candidate.opposite_nodes
    t0, t1 = candidate.triangles

    triangles = mesh.triangles.copy()
    triangles[t0] = (c, d, a)
    triangles[t1] = (d, c, b)
    return mesh.with_triangles(
        triangles,
        preserve_degree_targets=preserve_degree_targets)


def flip_edge(mesh, edge, preserve_degree_targets=True):
    """Return a new mesh obtained by a topology-only edge flip."""
    return flip_candidate(
        mesh,
        find_flip_candidate(mesh, edge),
        preserve_degree_targets=preserve_degree_targets)


__all__ = [
    "FlipCandidate",
    "canonical_edge",
    "degree_after_flip",
    "degree_l1_after_flip",
    "find_flip_candidate",
    "flip_candidate",
    "flip_candidates",
    "flip_edge",
    "flippable_edges",
    "mesh_edges",
]
