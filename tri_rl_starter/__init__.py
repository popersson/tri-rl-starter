"""Lean triangular-mesh improvement starter package."""

from .geometry import CASE_NAMES, generate_case_mesh
from .mesh import TriMesh
from .env import MeshCaseConfig, TriFlipEnv, TriFlipObservation

__all__ = [
    "CASE_NAMES",
    "MeshCaseConfig",
    "TriFlipEnv",
    "TriFlipObservation",
    "TriMesh",
    "generate_case_mesh",
]
