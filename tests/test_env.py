import numpy as np
import pytest

from tri_rl_starter.env import (
    NODE_FEATURE_NAMES,
    MeshCaseConfig,
    TriFlipEnv,
)
from tri_rl_starter.geometry import generate_case_mesh


def sample_mesh():
    return generate_case_mesh(case="random_polygon", seed=1, hmax=0.35)


def test_reset_observation_shapes_and_degree_features():
    mesh = sample_mesh()
    env = TriFlipEnv(mesh=mesh, max_steps=3)
    observation, info = env.reset()

    assert observation.node_features.shape == (
        mesh.npoints, len(NODE_FEATURE_NAMES))
    assert observation.edge_index.shape[0] == 2
    assert observation.candidate_edges.shape[1] == 2
    assert observation.action_mask.shape == (observation.n_candidates + 1,)
    assert observation.pos.shape == mesh.points.shape
    np.testing.assert_allclose(observation.node_features[:, 0], mesh.degrees())
    np.testing.assert_allclose(
        observation.node_features[:, 1], mesh.ideal_degrees)
    np.testing.assert_allclose(
        observation.node_features[:, 2],
        mesh.degrees() - mesh.ideal_degrees)
    assert info["degree_l1"] == pytest.approx(observation.degree_l1)


def test_noop_has_zero_reward_and_can_truncate():
    env = TriFlipEnv(mesh=sample_mesh(), max_steps=1)
    observation, _ = env.reset()
    before = observation.degree_l1

    observation, reward, terminated, truncated, info = env.step(0)

    assert reward == pytest.approx(0.0)
    assert observation.degree_l1 == pytest.approx(before)
    assert not terminated
    assert truncated
    assert info["action"] == 0
    assert info["flipped_edge"] is None


def test_flip_step_reward_matches_score_drop():
    env = TriFlipEnv(mesh=sample_mesh(), max_steps=3)
    observation, _ = env.reset()
    before = observation.score

    observation, reward, terminated, truncated, info = env.step(1)

    assert reward == pytest.approx(before - observation.score)
    assert info["flipped_edge"] is not None
    assert observation.step_count == 1
    assert not truncated
    assert isinstance(terminated, bool)


def test_invalid_action_raises():
    env = TriFlipEnv(mesh=sample_mesh(), max_steps=3)
    observation, _ = env.reset()

    with pytest.raises(ValueError, match="outside"):
        env.step(observation.n_actions)


def test_reset_restores_fixed_mesh():
    mesh = sample_mesh()
    env = TriFlipEnv(mesh=mesh, max_steps=3)
    observation0, _ = env.reset()
    env.step(1)
    observation1, _ = env.reset()

    np.testing.assert_array_equal(env.mesh.triangles, mesh.triangles)
    assert observation1.degree_l1 == pytest.approx(observation0.degree_l1)


def test_seeded_generated_reset_is_deterministic():
    config = MeshCaseConfig(
        case="random_polygon",
        nvertices=(10, 12),
        hmax=0.35,
    )
    env = TriFlipEnv(mesh_config=config, max_steps=3)
    observation0, _ = env.reset(seed=7)
    points0 = env.mesh.points.copy()
    triangles0 = env.mesh.triangles.copy()
    candidates0 = observation0.candidate_edges.copy()

    observation1, _ = env.reset(seed=7)

    np.testing.assert_allclose(env.mesh.points, points0)
    np.testing.assert_array_equal(env.mesh.triangles, triangles0)
    np.testing.assert_array_equal(observation1.candidate_edges, candidates0)
