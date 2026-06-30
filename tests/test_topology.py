import numpy as np
import pytest

from tri_rl_starter.geometry import generate_case_mesh
from tri_rl_starter.mesh import TriMesh, _edge_counts
from tri_rl_starter.topology import (
    degree_after_flip,
    degree_l1_after_flip,
    find_flip_candidate,
    flip_edge,
    flippable_edges,
)


def sample_mesh():
    return generate_case_mesh(case="random_polygon", seed=1, hmax=0.35)


def test_flippable_edges_are_interior_and_have_unique_new_diagonal():
    mesh = sample_mesh()
    counts = _edge_counts(mesh.triangles)
    edge_set = set(counts)

    for edge in flippable_edges(mesh):
        edge = tuple(edge)
        candidate = find_flip_candidate(mesh, edge)
        assert len(counts[edge]) == 2
        assert candidate.new_edge not in edge_set


def test_flip_preserves_mesh_size_boundary_and_degree_targets():
    mesh = sample_mesh()
    edge = flippable_edges(mesh)[0]
    candidate = find_flip_candidate(mesh, edge)
    flipped = flip_edge(mesh, edge)

    assert flipped.npoints == mesh.npoints
    assert flipped.nelements == mesh.nelements
    np.testing.assert_array_equal(flipped.boundary_edges, mesh.boundary_edges)
    np.testing.assert_array_equal(flipped.corner_nodes, mesh.corner_nodes)
    np.testing.assert_allclose(flipped.ideal_degrees, mesh.ideal_degrees)
    np.testing.assert_array_equal(flipped.minimum_degrees, mesh.minimum_degrees)

    flipped_edges = set(_edge_counts(flipped.triangles))
    assert candidate.edge not in flipped_edges
    assert candidate.new_edge in flipped_edges
    np.testing.assert_array_equal(flipped.degrees(), degree_after_flip(mesh, edge))


def test_local_degree_l1_after_flip_matches_full_recompute():
    mesh = sample_mesh()
    for edge in flippable_edges(mesh)[:25]:
        flipped = flip_edge(mesh, edge)
        assert degree_l1_after_flip(mesh, edge) == pytest.approx(
            flipped.degree_deviation()["l1"])


def test_boundary_edge_is_not_flippable():
    mesh = sample_mesh()
    counts = _edge_counts(mesh.triangles)
    boundary_edge = next(edge for edge, adjacent in counts.items()
                         if len(adjacent) == 1)

    with pytest.raises(ValueError, match="not a valid topological flip"):
        flip_edge(mesh, boundary_edge)


def test_duplicate_new_diagonal_is_rejected():
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 1.5],
    ])
    triangles = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 4],
    ])
    mesh = TriMesh(points, triangles, boundary_edges=np.empty((0, 2), dtype=int))

    with pytest.raises(ValueError, match="not a valid topological flip"):
        flip_edge(mesh, (0, 1))
