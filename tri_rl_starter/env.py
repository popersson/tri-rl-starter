"""Gym-style topology-only edge-flip environment."""

from dataclasses import dataclass
import math

import numpy as np

from .geometry import generate_case_mesh
from .topology import flip_candidate, flip_candidates, mesh_edges


NODE_FEATURE_NAMES = (
    "degree",
    "ideal_degree",
    "signed_degree_error",
    "abs_degree_error",
    "boundary_flag",
    "corner_flag",
    "minimum_degree",
    "minimum_degree_deficit",
)


@dataclass(frozen=True)
class MeshCaseConfig:
    """Arguments used to generate a fresh triangular mesh on reset."""

    case: str = "random_polygon"
    seed: int | None = None
    nvertices: int | tuple[int, int] = 20
    nblocks: int = 25
    np_out: int = 20
    np_in: int = 8
    chamfer_probability: float = 0.4
    inner_scale: float = 1.0 / 3.0
    hmax: float | None = 0.35
    min_angle: float = 28.6
    quality: bool = True


@dataclass(frozen=True)
class TriFlipObservation:
    """Array observation returned by ``TriFlipEnv``."""

    node_features: np.ndarray
    edge_index: np.ndarray
    candidate_edges: np.ndarray
    action_mask: np.ndarray
    pos: np.ndarray
    degree: np.ndarray
    ideal_degree: np.ndarray
    minimum_degree: np.ndarray
    score: float
    degree_l1: float
    optimum_score: float
    step_count: int
    max_steps: int

    @property
    def n_actions(self):
        return int(len(self.action_mask))

    @property
    def n_candidates(self):
        return int(len(self.candidate_edges))

    def as_dict(self):
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "candidate_edges": self.candidate_edges,
            "action_mask": self.action_mask,
            "pos": self.pos,
            "degree": self.degree,
            "ideal_degree": self.ideal_degree,
            "minimum_degree": self.minimum_degree,
            "score": self.score,
            "degree_l1": self.degree_l1,
            "optimum_score": self.optimum_score,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }


def degree_l1_score(mesh):
    return float(mesh.degree_deviation()["l1"])


def optimum_degree_l1_score(mesh):
    return float(abs(mesh.degree_deviation()["signed"].sum()))


def min_degree_deficit_total(mesh):
    return float(mesh.minimum_degree_deficit()["total"])


def objective_score(mesh, min_degree_weight=0.0):
    return (
        degree_l1_score(mesh)
        + float(min_degree_weight) * min_degree_deficit_total(mesh)
    )


def node_features(mesh, degrees=None):
    """Return degree-only node features for the v1 graph policy."""
    if degrees is None:
        degrees = mesh.degrees()
    degrees = np.asarray(degrees, dtype=float)
    ideal = mesh.ideal_degrees.astype(float)
    signed = degrees - ideal
    absolute = np.abs(signed)
    boundary = np.zeros(mesh.npoints, dtype=float)
    boundary[np.asarray(mesh.boundary_nodes, dtype=int)] = 1.0
    corner = np.zeros(mesh.npoints, dtype=float)
    corner[np.asarray(mesh.corner_nodes, dtype=int)] = 1.0
    minimum = mesh.minimum_degrees.astype(float)
    min_deficit = np.maximum(minimum - degrees, 0.0)
    return np.column_stack((
        degrees,
        ideal,
        signed,
        absolute,
        boundary,
        corner,
        minimum,
        min_deficit,
    ))


def normalized_positions(points):
    """Return centered and scale-normalized coordinates for metadata/plotting."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points.reshape(0, 2)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    scale = float(np.max(hi - lo))
    if scale <= 0.0 or not math.isfinite(scale):
        scale = 1.0
    return (points - center) / scale


def _edge_index(mesh):
    edges = mesh_edges(mesh, directed=True)
    if len(edges) == 0:
        return np.empty((2, 0), dtype=int)
    return edges.T.astype(int, copy=False)


def _sample_nvertices(nvertices, rng):
    if isinstance(nvertices, (tuple, list)):
        if len(nvertices) != 2:
            raise ValueError("nvertices range must be a pair")
        lo, hi = (int(nvertices[0]), int(nvertices[1]))
        if lo > hi:
            raise ValueError("nvertices range lower bound exceeds upper bound")
        return int(rng.integers(lo, hi + 1))
    return int(nvertices)


def _generate_mesh(config, rng):
    seed = (
        int(config.seed)
        if config.seed is not None
        else int(rng.integers(1, 2**31 - 1))
    )
    return generate_case_mesh(
        case=config.case,
        seed=seed,
        nvertices=_sample_nvertices(config.nvertices, rng),
        nblocks=config.nblocks,
        np_out=config.np_out,
        np_in=config.np_in,
        chamfer_probability=config.chamfer_probability,
        inner_scale=config.inner_scale,
        hmax=config.hmax,
        min_angle=config.min_angle,
        quality=config.quality,
    )


class TriFlipEnv:
    """Topology-only triangular edge-flip environment.

    Action ``0`` is a no-op. Action ``k + 1`` flips
    ``observation.candidate_edges[k]`` from the current observation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mesh=None,
        mesh_config=None,
        max_steps=None,
        max_steps_factor=1.0,
        min_degree_weight=0.0,
    ):
        self.fixed_mesh = mesh
        self.mesh_config = (
            MeshCaseConfig() if mesh_config is None else mesh_config
        )
        self.requested_max_steps = max_steps
        self.max_steps_factor = float(max_steps_factor)
        self.min_degree_weight = float(min_degree_weight)
        self.mesh = None
        self.step_count = 0
        self.max_steps = 0
        self.initial_score = 0.0
        self.best_score = 0.0
        self._observation = None
        self._candidate_objects = ()
        self._degrees = None
        self._degree_l1 = 0.0
        self._optimum_score = 0.0
        self.reset()

    def _invalidate_observation(self):
        self._observation = None

    def _set_degrees(self, degrees):
        self._degrees = np.asarray(degrees, dtype=int).copy()
        signed = self._degrees.astype(float) - self.mesh.ideal_degrees
        self._degree_l1 = float(np.abs(signed).sum())
        self._optimum_score = float(abs(signed.sum()))

    def _min_degree_deficit_total(self):
        deficit = np.maximum(self.mesh.minimum_degrees - self._degrees, 0)
        return float(deficit.sum())

    def _score_from_cache(self):
        return (
            self._degree_l1
            + self.min_degree_weight * self._min_degree_deficit_total()
        )

    def _update_degrees_for_flip(self, candidate):
        degrees = self._degrees.copy()
        a, b = candidate.edge
        c, d = candidate.opposite_nodes
        degrees[[a, b]] -= 1
        degrees[[c, d]] += 1
        self._set_degrees(degrees)

    def reset(self, seed=None, mesh=None):
        rng = np.random.default_rng(seed)
        if mesh is not None:
            self.mesh = mesh
        elif self.fixed_mesh is not None:
            self.mesh = self.fixed_mesh
        else:
            self.mesh = _generate_mesh(self.mesh_config, rng)

        self.step_count = 0
        self._invalidate_observation()
        if self.requested_max_steps is None:
            self.max_steps = max(
                1,
                int(math.ceil(self.max_steps_factor * self.mesh.nelements)))
        else:
            self.max_steps = int(self.requested_max_steps)
            if self.max_steps < 1:
                raise ValueError("max_steps must be positive")

        self._set_degrees(self.mesh.degrees())
        observation = self.observation()
        self.initial_score = observation.score
        self.best_score = self.initial_score
        return observation, self.info()

    def observation(self):
        if self._observation is not None:
            return self._observation

        candidates = flip_candidates(self.mesh)
        self._candidate_objects = candidates
        candidate_edges = (
            np.asarray([candidate.edge for candidate in candidates], dtype=int)
            if candidates else np.empty((0, 2), dtype=int)
        )
        mask = np.ones(len(candidate_edges) + 1, dtype=bool)
        features = node_features(self.mesh, degrees=self._degrees)
        degrees = features[:, 0].copy()
        ideal = features[:, 1].copy()
        minimum = features[:, 6].copy()
        self._observation = TriFlipObservation(
            node_features=features,
            edge_index=_edge_index(self.mesh),
            candidate_edges=candidate_edges,
            action_mask=mask,
            pos=normalized_positions(self.mesh.points),
            degree=degrees,
            ideal_degree=ideal,
            minimum_degree=minimum,
            score=self._score_from_cache(),
            degree_l1=self._degree_l1,
            optimum_score=self._optimum_score,
            step_count=self.step_count,
            max_steps=self.max_steps,
        )
        return self._observation

    def info(self):
        observation = self.observation()
        min_deficit = observation.node_features[:, 7]
        return {
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "score": observation.score,
            "degree_l1": observation.degree_l1,
            "optimum_score": observation.optimum_score,
            "min_degree_deficit_total": int(min_deficit.sum()),
            "min_degree_deficit_count": int(np.count_nonzero(min_deficit)),
            "best_score": self.best_score,
        }

    def step(self, action):
        action = int(action)
        observation = self.observation()
        if action < 0 or action >= observation.n_actions:
            raise ValueError(
                f"action {action} is outside [0, {observation.n_actions})")

        score_before = observation.score
        flipped_edge = None
        if action != 0:
            candidate = self._candidate_objects[action - 1]
            flipped_edge = candidate.edge
            self.mesh = flip_candidate(self.mesh, candidate)
            self._update_degrees_for_flip(candidate)

        self.step_count += 1
        self._invalidate_observation()
        next_observation = self.observation()
        self.best_score = min(self.best_score, next_observation.score)
        reward = float(score_before - next_observation.score)
        terminated = bool(next_observation.degree_l1 <= next_observation.optimum_score)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        info = self.info()
        info["action"] = action
        info["flipped_edge"] = flipped_edge
        return next_observation, reward, terminated, truncated, info


__all__ = [
    "MeshCaseConfig",
    "NODE_FEATURE_NAMES",
    "TriFlipEnv",
    "TriFlipObservation",
    "degree_l1_score",
    "min_degree_deficit_total",
    "node_features",
    "normalized_positions",
    "objective_score",
    "optimum_degree_l1_score",
]
